import uvicorn
from fastapi import FastAPI
from pydantic import BaseModel
import sys, os

# Add src directory to path to import local modules
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
print(f"Project root set to: {PROJECT_ROOT}")
sys.path.append(PROJECT_ROOT)


# Import core LLMOps components (ensure these files are ready)
from artifacts.src.models import load_llm 
from artifacts.src.ingestion import create_vector_db # Use the load-or-create function
from artifacts.src.graph import workflow # Import the StateGraph instance

# --- 1. Define Request/Response Schemas ---
class MatchRequest(BaseModel):
    """Schema for the input data."""
    resume_text: str
    job_description: str

class MatchResponse(BaseModel):
    """Schema for the output data."""
    match_score: str
    analysis: str
    
# --- 2. Initialize Resources (Runs once at startup) ---
# NOTE: This ensures the LLM and Vector DB are loaded *before* the API starts
print("--- Initializing FastAPI Resources ---")
try:
    # 2a. Load Vector DB
    vectorstore = create_vector_db()
    
    # 2b. Load LLM (handles Mac/Cloud detection)
    llm = load_llm()
    
    # 2c. Compile LangGraph
    # We pass necessary resources to the graph (if needed, or assume they are global/passed via nodes)
    # For simplicity, we assume the graph code directly uses 'vectorstore' and 'llm' from scope.
    app = workflow.compile()
    
    print("--- Initialization Complete ---")

except Exception as e:
    print(f"FATAL ERROR during startup: {e}")
    # Exit gracefully if resources fail to load
    sys.exit(1)

# --- 3. Define FastAPI App ---
app_fastapi = FastAPI(title="LLMOps Resume Matcher API")

@app_fastapi.post("/match", response_model=MatchResponse)
async def match_resume(request: MatchRequest):
    """
    Accepts resume and job description, runs them through the LangGraph pipeline,
    and returns the match analysis and score.
    """
    
    # Prepare the input state for the LangGraph
    input_state = {
        "resume_text": request.resume_text,
        "job_description": request.job_description,
    }
    
    # 3a. Invoke the LangGraph pipeline
    # NOTE: The LangGraph workflow must be defined such that it has access to 
    # the retriever created from the loaded vectorstore.
    
    # For simplicity here, we assume the retrieve_node function is updated 
    # to use the globally defined 'vectorstore' or 'retriever' object.
    
    final_state = app.invoke(input_state)
    
    # 3b. Extract results
    score = final_state.get("match_score", "N/A")
    analysis_text = final_state.get("analysis", "Analysis failed.")
    
    # 3c. Return structured response
    return MatchResponse(
        match_score=score,
        analysis=analysis_text
    )

# --- 4. Running the API ---
if __name__ == "__main__":
    # In production/Docker, this runs on port 80/8000
    uvicorn.run(app_fastapi, host="0.0.0.0", port=8000)