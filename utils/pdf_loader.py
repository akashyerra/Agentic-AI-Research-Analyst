import os
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document

def process_pdf(file_path: str) -> list[Document]:
    """
    Loads a PDF, extracts the text, and splits it into manageable, overlapping chunks.
    """
    print(f"--- RAG SYSTEM: Processing Document {file_path} ---")
    
    loader = PyPDFLoader(file_path)
    raw_documents = loader.load()
    
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len,
        is_separator_regex=False,
    )
    
    chunked_documents = text_splitter.split_documents(raw_documents)
    
    print(f"    -> Extracted {len(chunked_documents)} chunks.")
    return chunked_documents