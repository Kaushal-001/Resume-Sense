from typing import TypedDict
from langgraph.graph import StateGraph, END, START
from langchain_core.prompts import PromptTemplate
from sklearn.metrics.pairwise import cosine_similarity
from langchain.embeddings import HuggingFaceEmbeddings # Required for embed_model
import numpy as np
import re

import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from artifacts.src.models import load_llm
from artifacts.src.ingestion import create_vector_db

# --- Helper Functions (Provided by User, adapted for imports) ---
def clean_text(text):
    text = re.sub(r'\s+', ' ', text)
    return text.strip().lower()

def compute_match_score(resume_text, job_text, embedding_model, vectorstore, top_k=5):
    """Calculates baseline score, retrieves context, and finds missing skills heuristically."""
    # Clean inputs
    resume_text = clean_text(resume_text)
    job_text = clean_text(job_text)

    # Embed resume and job
    # NOTE: This assumes embedding_model is a global object that can run .embed_query()
    resume_emb = embedding_model.embed_query(resume_text)
    job_emb = embedding_model.embed_query(job_text)

    # Similarity score between full resume and job
    base_similarity = cosine_similarity([resume_emb], [job_emb])[0][0]

    # Also retrieve top matching chunks from FAISS for deeper analysis
    # NOTE: vectorstore needs to be the Langchain FAISS vectorstore object
    similar_chunks = vectorstore.similarity_search(job_text, k=top_k)

    # Analyze missing skills (basic heuristic)
    # Use simple word finding for robustness
    job_keywords = set(re.findall(r'\b[A-Za-z]{4,}\b', job_text)) # Min length 4
    resume_keywords = set(re.findall(r'\b[A-Za-z]{4,}\b', resume_text))
    
    # Filter for skills in job but not in resume
    missing_skills = list(job_keywords - resume_keywords)
    
    # Simple list of top 15 missing words (LLM will refine this)
    missing_skills_str = ", ".join(missing_skills[:15])

    match_percent = round(float(base_similarity) * 100, 2)

    return {
        "baseline_score": str(int(match_percent)), # Ensure integer for prompt clarity
        "retrived_docs": similar_chunks,
        "missing_skills_list": missing_skills_str
    }
# -----------------------------------------------------------------

# 1. Define State (Updated to align with new scoring function output)
class AgentState(TypedDict):
    resume_text: str
    job_description: str
    retrived_docs: list
    analysis: str
    match_score: str
    baseline_score: str       # New: Heuristic score string (e.g., '65')
    missing_skills_list: str  # New: Comma-separated list of keywords

# 2. Initialize Resources (Assumes embedding model loading)
llm = load_llm()
vectorstore = create_vector_db()
# Load embedding model globally as required by the new function
try:
    embedding_model = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
except Exception as e:
    # Handle this error based on your environment (e.g., set to None if using an external service)
    print(f"FATAL: Error loading embedding model: {e}")
    sys.exit(1)


# 3. Define Nodes
def score_and_retrieve_node(state: AgentState):
    """Calculate match score, retrieve chunks, and find missing skills heuristically."""

    print("🔍 Calculating score and retrieving context...")
    
    results = compute_match_score(
        resume_text=state["resume_text"], 
        job_text=state["job_description"], 
        embedding_model=embedding_model, 
        vectorstore=vectorstore
    )
    
    # Store all results in the state
    return {
        "retrived_docs": results["retrived_docs"],
        "baseline_score": results["baseline_score"],
        "match_score": results["baseline_score"], # Set as default fallback
        "missing_skills_list": results["missing_skills_list"]
    }


def analysis_node(state: AgentState):
    """Analyze the resume, recommend improvements based on the heuristic skill list."""
    
    print("📝 Analyzing resume and generating recommendations...")
    
    # Inject the heuristic findings into the prompt
    baseline_score = state["baseline_score"]
    missing_skills = state["missing_skills_list"]
    
    prompt = PromptTemplate(
        input_variables=["resume_text", "job_description", "retrived_docs", "baseline_score", "missing_skills"],
        template=f"""
        You are an expert AI Hiring Evaluator. Analyze the candidate's resume against the job description,
        using the retrieved documents as supporting context.

        The **BASELINE VECTOR SCORE** for the full documents is: {{baseline_score}}

        A **HEURISTIC ANALYSIS** suggests the following keywords are in the job description but MISSING from the resume:
        {{missing_skills}}

        Resume:
        {{resume_text}}

        Job Description:
        {{job_description}}

        Relevant Documents (Context for specific job requirements):
        {{retrived_docs}}

        Your output MUST follow this structure precisely:

        ===========================
        📌 **DETAILED ANALYSIS**
        - Provide a clear, deep comparison between resume and JD.
        - Discuss strengths, alignment areas, and major gaps (confirming the heuristic list).

        ===========================
        📌 **ESSENTIAL_SKILLS_MISSING**
        List the MOST IMPORTANT skills missing. Refine the heuristic list provided above ({{missing_skills}}) by removing generic words and focusing on **technical or domain-specific terms** required by the JD.
        - Use short, numbered bullet points (1., 2., 3., etc.).

        ===========================
        📌 **RECOMMENDATIONS_TO_IMPROVE**
        Provide actionable steps to make the candidate a better fit.
        **Focus on the missing skills (from the list above) and the job's context.**
        - For each missing skill, suggest a concrete project, framework, or area of study.
        - Example: To gain skill 'RAG Pipelines', suggest 'Build a RAG system using LangChain, FAISS, and a local LLM, focusing on error handling and latency optimization.'
        - Use short, numbered bullet points.

        ===========================
        📌 **FINAL MATCH SCORE**
        Adjust the baseline score ({{baseline_score}}) based on your comprehensive analysis and specific details found in the resume. 
        
        At the end of your output, on a NEW line, include:

        MATCH_SCORE_FINAL: <score>

        RULES:
        - Score must be an integer 0–100
        - Do NOT include a % sign
        - The line MUST appear exactly once
        """
    )
    
    response = llm.generate([prompt.format(
        resume_text=state["resume_text"],
        job_description=state["job_description"],
        retrived_docs="\n".join([doc.page_content for doc in state["retrived_docs"]]),
        baseline_score=baseline_score,
        missing_skills=missing_skills
    )])
    
    analysis = response.generations[0][0].text
    
    # Attempt to extract LLM-adjusted score
    score_match = re.search(r"MATCH_SCORE_FINAL:\s*(\d+)", analysis)

    if score_match:
        match_score = score_match.group(1)
    else:
        # Fallback to the reliable calculated score
        match_score = state["match_score"] 
        print(f"⚠️  Warning: MATCH_SCORE_FINAL not found. Falling back to calculated score: {match_score}")

    return {
        "analysis": analysis,
        "match_score": match_score
    }

# 4. Build State Graph
workflow = StateGraph(AgentState)


# Add Nodes
workflow.add_node('score_and_retrieve', score_and_retrieve_node)
workflow.add_node('analysis', analysis_node)

# Add Edges
workflow.add_edge(START, "score_and_retrieve")
workflow.add_edge("score_and_retrieve", "analysis")
workflow.add_edge("analysis", END)

# 5. Compile
app = workflow.compile()