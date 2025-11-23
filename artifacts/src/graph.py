from typing import TypedDict
from langgraph.graph import StateGraph, END, START
from langchain_core.prompts import PromptTemplate

import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from artifacts.src.models import load_llm
from artifacts.src.ingestion import create_vector_db
import re

# 1. Define State
class AgentState(TypedDict):
    resume_text: str
    job_description: str
    retrived_docs: list
    analysis: str
    match_score: str

# 2. Initialize Resources
llm = load_llm()
vectorstore = create_vector_db()

# 3. Define Nodes
def retrieve_node(state: AgentState):
    """Retrieve relevant documents from vector store based on resume and job description."""

    print("🔍 Retrieving relevant documents...")
    query = state["job_description"]
    retriever = vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": 5})
    docs = retriever.invoke(query)
    return {"retrived_docs": docs}

def analysis_node(state: AgentState):
    """Analyze the resume against the job description using retrieved documents."""
    
    print("📝 Analyzing resume against job description...")
    prompt = PromptTemplate(
        input_variables=["resume_text", "job_description", "retrived_docs"],
        template="""
        You are an expert AI Hiring Evaluator. Analyze the candidate's resume against the job description,
        using the retrieved documents as supporting context.

        Resume:
        {resume_text}

        Job Description:
        {job_description}

        Relevant Documents:
        {retrived_docs}

        Your output MUST follow this structure:

        ===========================
        📌 **DETAILED ANALYSIS**
        - Provide a clear, deep comparison between resume and JD.
        - Discuss strengths, alignment areas, and gaps.

        ===========================
        📌 **ESSENTIAL_SKILLS_MISSING**
        List the MOST IMPORTANT skills missing from the resume that the JD requires.
        - Use short bullet points
        - Include both technical + soft skills if relevant

        ===========================
        📌 **RECOMMENDATIONS_TO_IMPROVE**
        Provide actionable steps to make the candidate a better fit.
        Examples:
        - Build projects in X
        - Learn frameworks A, B, C
        - Add measurable metrics to resume
        - Strengthen domain knowledge (FinTech, Healthcare, RAG, etc.)

        Be very specific and tailored.

        ===========================
        📌 **FINAL MATCH SCORE**
        At the end of your output, on a NEW line, include:

        MATCH_SCORE_FINAL: <score>

        RULES:
        - Score must be an integer 0–100
        - Do NOT include a % sign
        - The line MUST appear exactly once
        - Example: MATCH_SCORE_FINAL: 87
        """
    )
    response = llm.generate([prompt.format(
        resume_text=state["resume_text"],
        job_description=state["job_description"],
        retrived_docs="\n".join([doc.page_content for doc in state["retrived_docs"]])
    )])
    
    analysis = response.generations[0][0].text
    # Extract match score from analysis (mocked here for brevity)
    score_match = re.search(r"MATCH_SCORE_FINAL:\s*(\d+)", analysis)

    if score_match:
        match_score = score_match.group(1)
    else:
        match_score = "N/A"
        print("⚠️  Warning: MATCH_SCORE_FINAL not found in the analysis output.")

    return {
        "analysis": analysis,
        "match_score": match_score
    }

# 4. Build State Graph
workflow = StateGraph(AgentState)


# Add Nodes
workflow.add_node('retrieve',retrieve_node)
workflow.add_node('analysis',analysis_node)

# Add Edges
workflow.add_edge(START, "retrieve")
workflow.add_edge("retrieve", "analysis")
workflow.add_edge("analysis", END)

# 5. Compile
app = workflow.compile()

# ==========================================
# 5. GENERATE DIAGRAM
# ==========================================
from IPython.display import Image, display

try:
    # This generates a PNG of the graph structure
    display(Image(app.get_graph().draw_mermaid_png()))
except Exception as e:
    print("Could not generate diagram. Ensure you have 'langgraph' installed.")
    print(f"Error: {e}")
