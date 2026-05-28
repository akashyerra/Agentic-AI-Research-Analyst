from langchain_huggingface import HuggingFaceEmbeddings

def get_embeddings():
    """
    Initializes and returns the HuggingFace embedding model.
    Isolated here so we can easily swap to OpenAI or Cohere later if needed.
    """
    return HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")