import streamlit as st
import requests
import re

# Base URL for FastAPI backend
API_URL = "http://127.0.0.1:8000/match"

# --- UI CONFIG ---
st.set_page_config(layout="wide")
st.title("🤖 Resume–Job Match Analyzer (Client UI)")
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

                    analysis = data.get("analysis", "")
                    score = data.get("match_score", "N/A")

                    # ========== Extract new sections from analysis ==========
                    # These sections were added in your enhanced analysis_node

                    # --- Extract ESSENTIAL_SKILLS_MISSING ---
                    skills_missing_match = re.search(
                        r"ESSENTIAL_SKILLS_MISSING[\s\S]*?==========================",
                        analysis
                    )
                    missing_skills = skills_missing_match.group(0) if skills_missing_match else "Not found."

                    # --- Extract RECOMMENDATIONS_TO_IMPROVE ---
                    recommendations_match = re.search(
                        r"RECOMMENDATIONS_TO_IMPROVE[\s\S]*?==========================",
                        analysis
                    )
                    recommendations = recommendations_match.group(0) if recommendations_match else "Not found."

                    # Clean the extracted block endings
                    missing_skills = re.sub(r"==========================", "", missing_skills).strip()
                    recommendations = re.sub(r"==========================", "", recommendations).strip()

                    # --- DISPLAY MATCH SCORE ---
                    st.subheader("🎯 Match Score")
                    if str(score).isdigit():
                        st.metric("Match Percentage", f"{score}%")
                    else:
                        st.info(f"Score: {score}")

                    # --- DISPLAY DETAILED ANALYSIS ---
                    st.subheader("📊 Detailed Analysis")
                    st.markdown(analysis)

                    # --- Display Extracted Sections ---
                    st.markdown("---")

                    with st.expander("❗ Essential Skills Missing"):
                        st.markdown(missing_skills)

                    with st.expander("🔧 Recommendations to Improve Fit"):
                        st.markdown(recommendations)

                else:
                    st.error(f"API Error: {response.status_code}")
                    st.json(response.json())

            except requests.exceptions.ConnectionError:
                st.error("🚫 Could not connect to FastAPI. Ensure 'api.py' is running on port 8000.")
            except Exception as e:
                st.error(f"Unexpected Error: {e}")

# Required for Streamlit
if __name__ == "__main__":
    pass
