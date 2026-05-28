def retrieve_context(query: str, vector_store, top_k: int = 3) -> str:
    """
    Executes a semantic search against the provided vector database.
    Formats the retrieved chunks into a single readable string for the LLM.
    """
    results = vector_store.similarity_search(query, k=top_k)
    
    context_data = "\n\n".join([doc.page_content for doc in results])
    return context_data