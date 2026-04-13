# phase1_rag/embed_webmd_data.py

import os
import pandas as pd
import chromadb
from chromadb.utils import embedding_functions
from langchain_text_splitters import RecursiveCharacterTextSplitter
from tqdm import tqdm # pip install tqdm (optional, for progress bars)

# Paths
WIKIDOC_PATH = "data/raw/medical_meadow_wikidoc.csv"
CHROMA_DB_PATH = "./chroma_db"

def build_vector_db():
    print("1. Loading dataset...")
    # Load Wikidoc (columns: 'instruction', 'input', 'output')
    df = pd.read_csv(WIKIDOC_PATH)
    df = df.dropna(subset=['output'])
    
    # Combine the topic ('instruction') with the factual text ('output')
    df['text'] = "Topic: " + df['instruction'].astype(str) + "\nInformation: " + df['output'].astype(str)
    
    print("2. Initializing ChromaDB and Text Splitter...")
    client = chromadb.PersistentClient(path=CHROMA_DB_PATH)
    
    # We use the all-MiniLM-L6-v2 model, which is fast and lightweight
    sentence_transformer_ef = embedding_functions.SentenceTransformerEmbeddingFunction(
        model_name="all-MiniLM-L6-v2"
    )
    
    collection = client.get_or_create_collection(
        name="pharma_knowledge_base",
        embedding_function=sentence_transformer_ef
    )
    
    # Text splitter to keep medical chunks coherent (512 characters)
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=512,
        chunk_overlap=64,
        separators=["\n\n", "\n", ".", " ", ""]
    )
    
    print("3. Chunking text...")
    documents = []
    metadatas =[]
    ids =[]
    
    # To keep this fast for our use case, we process a subset of 15,000 highly relevant rows.
    # You can increase this limit to len(df) to embed the entire dataset.
    for index, row in tqdm(df.head(15000).iterrows(), total=min(15000, len(df))):
        chunks = splitter.split_text(row['text'])
        for i, chunk in enumerate(chunks):
            documents.append(chunk)
            metadatas.append({"source": "wikidoc", "topic": str(row['instruction'])})
            ids.append(f"doc_{index}_chunk_{i}")
            
    print(f"4. Indexing {len(documents)} chunks into ChromaDB...")
    
    # Insert in batches to prevent memory overflow
    batch_size = 5000
    for i in tqdm(range(0, len(documents), batch_size)):
        collection.add(
            documents=documents[i:i+batch_size],
            metadatas=metadatas[i:i+batch_size],
            ids=ids[i:i+batch_size]
        )
        
    print("✅ Vector database built successfully! Data saved to ./chroma_db")

if __name__ == "__main__":
    build_vector_db()