
import os
from langchain_community.document_loaders import PyPDFLoader, DirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings

DATA_DIR = "data"
VECTOR_DIR = "vector_store"

def load_documents():
    loader = DirectoryLoader(DATA_DIR, glob="**/*.pdf", loader_cls=PyPDFLoader)
    docs = loader.load()
    print(f"Loaded {len(docs)} raw documents")
    return docs

def chunk_documents(docs):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=100
    )
    chunks = splitter.split_documents(docs)
    print(f"Split into {len(chunks)} chunks")
    return chunks

def build_vector_store(chunks):
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    vectordb = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=VECTOR_DIR
    )
    vectordb.persist()
    print("Vector store saved.")
    return vectordb, chunks

if __name__ == "__main__":
    docs = load_documents()
    chunks = chunk_documents(docs)
    build_vector_store(chunks)