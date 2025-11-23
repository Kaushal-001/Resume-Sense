import streamlit as st
import requests

# The base URL for your FastAPI server
# Since you run FastAPI on port 8000, this is the address:
API_URL = "http://127.0.0.1:8000/match"

# --- 1. Streamlit UI ---
st.set_page_config(layout="wide")
st.title("🤖 Resume & Job Matcher (Client)")
st.markdown("This app connects to a remote FastAPI AI service for analysis.")
st.markdown("---")

# --- 2. Input Fields ---
col1, col2 = st.columns(2)

with col1:
    resume_input = st.text_area("Candidate Resume Text", height=400, placeholder="Paste the candidate's full resume text here...")

with col2:
    job_input = st.text_area("Job Description Text", height=400, placeholder="Paste the full job description here...")

st.markdown("---")

# --- 3. Analysis Button & API Call ---
if st.button("Analyze Match via FastAPI", type="primary", use_container_width=True):
    if not resume_input or not job_input:
        st.warning("Please provide both Resume and Job Description text.")
    else:
        # Data payload matching the MatchRequest Pydantic model in api.py
        payload = {
            "resume_text": resume_input,
            "job_description": job_input,
        }
        
        with st.spinner("Sending data to FastAPI server for analysis..."):
            try:
                # Send the POST request to the API endpoint
                response = requests.post(API_URL, json=payload)
                
                # Check for successful response
                if response.status_code == 200:
                    # The response body is a JSON object matching the MatchResponse model
                    data = response.json()
                    
                    # --- 4. Display Results ---
                    analysis = data.get("analysis", "Analysis failed.")
                    score = data.get("match_score", "N/A")

                    st.subheader("Match Score")
                    
                    if score.isdigit():
                        st.metric(label="Match Percentage", value=f"{score}%")
                    else:
                        st.info(f"Score: {score}")

                    st.subheader("Detailed Analysis")
                    st.markdown(analysis)
                
                else:
                    st.error(f"API Error: Request failed with status code {response.status_code}.")
                    st.json(response.json()) # Display the error details if available

            except requests.exceptions.ConnectionError:
                st.error("Connection Error: Could not reach the FastAPI server. Ensure 'api.py' is running.")
            except Exception as e:
                st.error(f"An unexpected error occurred: {e}")

# --- 5. Running the App ---
if __name__ == "__main__":
    # Note: Streamlit runs the script from top to bottom on every user interaction!
    pass