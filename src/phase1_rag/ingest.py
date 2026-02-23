import os
import pandas as pd
from langchain_community.document_loaders import DataFrameLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma

# Define relative paths based on our folder structure
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../'))
DATA_PATH = os.path.join(BASE_DIR, 'data/raw/mtsamples.csv')
CHROMA_PATH = os.path.join(BASE_DIR, 'data/vector_db')

def main():
    print("🚀 Starting Phase 1: Medical Knowledge Base Ingestion")

    # 1. Load the Data
    print(f"Loading data from {DATA_PATH}...")
    df = pd.read_csv(DATA_PATH)
    
    # Drop rows where the actual medical transcription is missing
    df = df.dropna(subset=['transcription'])
    
    # We only need a subset for a fast test. Let's use 500 records.
    # (Remove the .head(500) later if you want to ingest all 5000+ records)
    df = df.head(500) 
    
    # Use LangChain's DataFrameLoader to treat the 'transcription' column as the main text
    loader = DataFrameLoader(df, page_content_column="transcription")
    documents = loader.load()
    print(f"Loaded {len(documents)} medical transcripts.")

    # 2. Chunk the Text
    # LLMs have context limits. We split long notes into 1000-character chunks with a 150-char overlap
    print("Chunking documents...")
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000, 
        chunk_overlap=150,
        length_function=len
    )
    chunks = text_splitter.split_documents(documents)
    print(f"Split transcripts into {len(chunks)} overlapping chunks.")

    # 3. Initialize the Embedding Model (Optimized for Apple Silicon 'mps')
    print("Loading HuggingFace Embedding Model (all-MiniLM-L6-v2) on MPS...")
    embeddings = HuggingFaceEmbeddings(
        model_name="all-MiniLM-L6-v2",
        model_kwargs={'device': 'mps'} # Hardware acceleration for Mac!
    )

    # 4. Create and Persist the Vector Database
    print(f"Generating embeddings and saving to ChromaDB at {CHROMA_PATH}...")
    db = Chroma.from_documents(
        documents=chunks, 
        embedding=embeddings, 
        persist_directory=CHROMA_PATH
    )
    
    print("✅ Ingestion Complete! Your ChromaDB is ready.")

    # --- QUICK TEST ---
    print("\n--- Running a Quick Retrieval Test ---")
    query = "Patient presents with severe chest pain and shortness of breath."
    results = db.similarity_search(query, k=2) # Fetch top 2 most relevant chunks
    
    for i, res in enumerate(results):
        print(f"\nResult {i+1}:")
        print(f"Text: {res.page_content[:200]}...") # Print first 200 chars
        print(f"Metadata: {res.metadata}")

if __name__ == "__main__":
    main()