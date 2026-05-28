from langchain_community.vectorstores import FAISS
from utils.pdf_loader import process_pdf
from rag.embeddings import get_embeddings

def create_vector_store(pdf_path: str, save_path: str = "data/faiss_index"):
    """
    Processes a PDF, converts chunks to vectors, and saves the FAISS index to disk.
    """
    chunks = process_pdf(pdf_path)
    embeddings = get_embeddings()
    
    print("--- RAG SYSTEM: Generating Embeddings and Building Vector DB ---")
    vector_store = FAISS.from_documents(chunks, embeddings)
    vector_store.save_local(save_path)
    
    print(f"    -> Vector DB saved to {save_path}")
    return vector_store

def load_vector_store(load_path: str = "data/faiss_index"):
    """
    Loads an existing FAISS index from disk.
    """
    try:
        embeddings = get_embeddings()
        vector_store = FAISS.load_local(load_path, embeddings, allow_dangerous_deserialization=True)
        return vector_store
    except Exception as e:
        print(f"Error loading vector store: {e}")
        return None