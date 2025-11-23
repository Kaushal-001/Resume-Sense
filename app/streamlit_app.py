import streamlit as st
import sys

# Add src directory to path to import local modules
import sys, os

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(PROJECT_ROOT)

# Import core LLMOps components
from artifacts.src.models import load_llm 
from artifacts.src.ingestion import create_vector_db
from artifacts.src.graph import workflow # Import the StateGraph instance

# --- 1. Initialize Resources (Cached for efficiency) ---
@st.cache_resource
def setup_pipeline():
    """Initializes and caches the heavy components (LLM, Vector DB)"""
    st.write("Initializing AI pipeline...")
    
    try:
        # Load Vector DB
        vectorstore = create_vector_db()
        
        # Load LLM
        llm = load_llm()
        
        # Compile LangGraph (assuming graph nodes use the loaded vectorstore/llm)
        app = workflow.compile()
        
        st.success("Pipeline Ready!")
        return app
    except Exception as e:
        st.error(f"Failed to load pipeline resources: {e}")
        st.stop()

# --- 2. Streamlit UI ---
st.set_page_config(layout="wide")
st.title("🤖 Resume & Job Matcher")
st.markdown("---")

# Setup the pipeline once
pipeline_app = setup_pipeline()

# --- 3. Input Fields ---
col1, col2 = st.columns(2)

with col1:
    resume_input = st.text_area("Candidate Resume Text", height=400, placeholder="Paste the candidate's full resume text here...")

with col2:
    job_input = st.text_area("Job Description Text", height=400, placeholder="Paste the full job description here...")

st.markdown("---")

# --- 4. Analysis Button ---
if st.button("Analyze Match", type="primary", use_container_width=True):
    if not resume_input or not job_input:
        st.warning("Please provide both Resume and Job Description text.")
    else:
        with st.spinner("Running AI Analysis..."):
            # Prepare the input state
            input_state = {
                "resume_text": resume_input,
                "job_description": job_input,
            }

            # Invoke the LangGraph pipeline
            final_state = pipeline_app.invoke(input_state)

        # --- 5. Display Results ---
        analysis = final_state.get("analysis", "Analysis failed.")
        score = final_state.get("match_score", "N/A")

        st.subheader("Match Score")
        
        # Display score as a large metric
        if score.isdigit():
            st.metric(label="Match Percentage", value=f"{score}%")
        else:
            st.info(f"Score: {score}")

        st.subheader("Detailed Analysis")
        st.markdown(analysis)

# --- 6. Running the App ---
if __name__ == "__main__":
    # Note: Streamlit runs the script from top to bottom on every user interaction!
    pass