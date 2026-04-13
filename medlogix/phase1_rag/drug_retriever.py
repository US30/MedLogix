# phase1_rag/drug_retriever.py

import chromadb
from chromadb.utils import embedding_functions

CHROMA_DB_PATH = "./chroma_db"

def retrieve_drug_context(query: str, top_k: int = 3) -> list[str]:
    """
    Retrieves the top_k relevant pharmacological context chunks for a given query.
    This will be used by the Agent to pull drug guidelines.
    """
    client = chromadb.PersistentClient(path=CHROMA_DB_PATH)
    
    # Must use the exact same embedding model used to build the DB
    sentence_transformer_ef = embedding_functions.SentenceTransformerEmbeddingFunction(
        model_name="all-MiniLM-L6-v2"
    )
    
    collection = client.get_collection(
        name="pharma_knowledge_base",
        embedding_function=sentence_transformer_ef
    )
    
    results = collection.query(
        query_texts=[query],
        n_results=top_k
    )
    
    # Extract and return the text documents from the results
    if results and results['documents']:
        return results['documents'][0]  # Return the list of strings for the first query
    return