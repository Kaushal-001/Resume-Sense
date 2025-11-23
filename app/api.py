import uvicorn
from fastapi import FastAPI
from pydantic import BaseModel
import sys, os

# --------------------------
# 1. Import Local Modules
# --------------------------
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
print(f"Project root set to: {PROJECT_ROOT}")
sys.path.append(PROJECT_ROOT)

from artifacts.src.models import load_llm
from artifacts.src.ingestion import create_vector_db
from artifacts.src.graph import workflow   # LangGraph workflow (StateGraph)


# --------------------------
# 2. Define Schemas
# --------------------------
class MatchRequest(BaseModel):
    resume_text: str
    job_description: str

class MatchResponse(BaseModel):
    match_score: str
    analysis: str


# --------------------------
# 3. Initialize Core Resources
# --------------------------
print("\n--- Initializing FastAPI + LangGraph Resources ---")

try:
    # Vector DB (FAISS / Chroma / etc.)
    vectorstore = create_vector_db()

    # LLM loader (Hermes, Phi-3, Mistral, etc.)
    llm = load_llm()

    # Compile LangGraph workflow into runnable graph app
    langgraph_app = workflow.compile()

    print("--- Initialization Complete ---\n")

except Exception as e:
    print(f"FATAL ERROR during startup: {e}")
    sys.exit(1)


# --------------------------
# 4. Create FastAPI App
# --------------------------
app_fastapi = FastAPI(
    title="LLMOps Resume Matcher API",
    description="FastAPI server running a LangGraph resume–JD matching pipeline",
    version="1.0.0",
)


# --------------------------
# 5. API Endpoint
# --------------------------
@app_fastapi.post("/match", response_model=MatchResponse)
async def match_resume(request: MatchRequest):
    """
    Sends resume & JD through the LangGraph pipeline.
    Returns:
    - analysis (string)
    - match_score (int as string)
    """

    # LangGraph requires state dictionary as input
    input_state = {
        "resume_text": request.resume_text,
        "job_description": request.job_description,
    }

    # Run the LangGraph DAG
    # NOTE: The graph internally uses your retrieve_node and analysis_node
    final_state = langgraph_app.invoke(input_state)

    # Extract results
    score = final_state.get("match_score", "N/A")
    analysis_text = final_state.get("analysis", "Analysis failed.")

    return MatchResponse(
        match_score=score,
        analysis=analysis_text
    )


# --------------------------
# 6. Run Server
# --------------------------
if __name__ == "__main__":
    uvicorn.run(app_fastapi, host="0.0.0.0", port=8000)
