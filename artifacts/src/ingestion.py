import pandas as pd
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
import os,sys

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
DATA_PATH = os.path.join(ROOT_DIR, "data")
print(f"Data path set to: {DATA_PATH}")

def create_vector_db():
    # 1. Load Data (Mocking your CSV load for brevity)
    # In production, load from S3 or a database
    jobs = pd.read_csv(os.path.join(DATA_PATH, "job_title_des.csv"))
    resumes = pd.read_csv(os.path.join(DATA_PATH, "Resume.csv"))
    
    # 2. Process Documents
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    docs = []
    for _,row in resumes.iterrows():
        text = f"The resume content is: {row['Resume_str']}. "
        metadata = {
            "type": "resume",
            "category": row["Category"],
        }
        chunks = text_splitter.split_text(text)
        for i, chunk in enumerate(chunks):
            docs.append(Document(
                page_content=chunk, 
                metadata={**metadata, "chunk_id": i}
            ))
    print(f"Processed {len(resumes)} resumes into {len(docs)} document chunks.")


    for _,row in jobs.iterrows():
        text = f"The job description is: {row['Job Description']}"
        metadata = {
            "type": "job",
            "job_title": row["Job Title"],
        }
        chunks = text_splitter.split_text(text)
        for i, chunk in enumerate(chunks):
            docs.append(Document(
                page_content=chunk, 
                metadata={**metadata, "chunk_id": i}
            ))
    
    print(f"Total documents created: {len(docs)}")

    # 3. Create Embeddings
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

    # 4. Create Vector Store
    vectorstore = FAISS.from_documents(docs, embeddings)

    # 5. Save Vector Store locally
    if not os.path.exists("artifacts"):
        os.makedirs("artifacts")
    vectorstore.save_local("artifacts/faiss_index")
    print("✅ FAISS Index created and saved.")
    return vectorstore

if __name__ == "__main__":
    create_vector_db()
