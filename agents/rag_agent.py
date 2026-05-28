import os
from graph.state import AgentState
from rag.vector_store import load_vector_store
from rag.retriever import retrieve_context

def retrieve_documents(state: AgentState) -> dict:
    """
    LangGraph node function that queries the local vector database
    for context related to the user's query.
    """
    print("--- RAG AGENT: Retrieving Internal Documents ---")
    
    query = state["query"]
    
    vector_store = load_vector_store()
    
    if vector_store is None:
        print("    -> No vector database found. Skipping document retrieval.")
        return {"retrieved_docs": "No internal documents uploaded or available."}
    
    print(f"    -> Searching for context related to: '{query}'")
    context = retrieve_context(query=query, vector_store=vector_store, top_k=3)
    
    if not context:
        print("    -> No relevant context found in documents.")
        return {"retrieved_docs": "Documents were searched, but no highly relevant context was found."}
        
    print("    -> Successfully retrieved document context.")
    
    return {"retrieved_docs": context}