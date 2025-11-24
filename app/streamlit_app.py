import streamlit as st
import requests
import re

# Base URL for FastAPI backend
API_URL = "http://127.0.0.1:8000/match"

# --- UI CONFIG ---
st.set_page_config(layout="wide")
st.title("🤖 Resume–Sense")
st.markdown("This Streamlit app connects to your FastAPI + LangGraph scoring engine.")
st.markdown("---")

# --- INPUT FIELDS ---
col1, col2 = st.columns(2)

with col1:
    resume_input = st.text_area(
        "📄 Candidate Resume",
        height=400,
        placeholder="Paste resume text here..."
    )

with col2:
    job_input = st.text_area(
        "📝 Job Description",
        height=400,
        placeholder="Paste job description here..."
    )

st.markdown("---")

# --- BUTTON: API CALL ---
if st.button("🔍 Analyze via FastAPI", type="primary", use_container_width=True):
    if not resume_input or not job_input:
        st.warning("Please provide both resume text and job description.")
    else:
        payload = {
            "resume_text": resume_input,
            "job_description": job_input,
        }

        with st.spinner("Sending data to FastAPI server for analysis..."):
            try:
                response = requests.post(API_URL, json=payload)

                if response.status_code == 200:
                    data = response.json()

                    analysis_raw = data.get("analysis", "")
                    score = data.get("match_score", "N/A")

                    # --- New Parsing Logic for Structured Output ---
                    
                    detailed_analysis = "N/A"
                    missing_skills = "Not found."
                    recommendations = "Not found."
                    
                    # 1. First, remove the final score line and the preceding delimiter
                    analysis_content = re.sub(
                        r'===========================\s*📌 \*\*FINAL MATCH SCORE\*\*[\s\S]*', 
                        '', 
                        analysis_raw
                    ).strip()

                    # 2. Split the remaining content using the section headers as delimiters
                    # Regex captures the header string itself (e.g., 'DETAILED ANALYSIS')
                    sections = re.split(r'===========================\s*📌 \*\*(.+?)\*\*', analysis_content)
                    
                    # Map the split sections (split results in [pre-header, header_title, content, header_title, content, ...])
                    parsed_sections = {}
                    for i in range(1, len(sections), 2):
                        header_key = sections[i].strip()
                        content = sections[i+1].strip()
                        parsed_sections[header_key] = content
                        
                    # 3. Assign content to display variables
                    detailed_analysis = parsed_sections.get('DETAILED ANALYSIS', 'Analysis failed to parse.')
                    missing_skills = parsed_sections.get('ESSENTIAL_SKILLS_MISSING', 'Could not parse skills missing.')
                    recommendations = parsed_sections.get('RECOMMENDATIONS_TO_IMPROVE', 'Could not parse recommendations.')


                    # --- DISPLAY MATCH SCORE ---
                    st.subheader("🎯 Match Score")
                    if str(score).isdigit():
                        st.metric("Match Percentage", f"{score}%")
                    else:
                        st.info(f"Score: {score}")
                        
                    st.markdown("---")

                    # --- DISPLAY STRUCTURED SECTIONS ---
                    
                    st.subheader("🔍 Detailed Analysis")
                    st.markdown(detailed_analysis)
                    
                    st.subheader("❌ Essential Skills Missing")
                    st.markdown(missing_skills)
                    
                    st.subheader("✅ Recommendations to Improve Fit")
                    st.markdown(recommendations)
                    
                    st.markdown("---")
                    with st.expander("View Full Raw LLM Output"):
                        st.text(analysis_raw)


                else:
                    st.error(f"API Error: {response.status_code}")
                    st.json(response.json())

            except requests.exceptions.ConnectionError:
                st.error("🚫 Could not connect to FastAPI. Ensure 'api.py' is running on port 8000.")
            except Exception as e:
                st.error(f"Unexpected Error during analysis: {e}")
                
# Required for Streamlit
if __name__ == "__main__":
    pass