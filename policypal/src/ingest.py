import os
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
# from langchain_openai import OpenAIEmbeddings
from langchain_ollama import OllamaEmbeddings
from langchain_community.vectorstores import Chroma
from dotenv import load_dotenv


# Load API Keys
load_dotenv()

# CONFIGURATION
DATA_PATH = "./data/data-science-from-scratch.pdf"
CHROMA_PATH = "./chroma_db"


def ingest_docs():
    # 1. LOAD
    print("Loading PDF...")
    loader = PyPDFLoader(DATA_PATH)
    raw_documents = loader.load()
    print(f"Loaded {len(raw_documents)} pages.")

    # 2. CHUNK (Critical Step)
    # We use a 1000 character window with 200 character overlap
    # to ensure context isn't lost between splits.
    print("Splitting text...")
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        add_start_index=True,  # Helps us trace where text came from
    )
    chunks = text_splitter.split_documents(raw_documents)
    print(f"Split into {len(chunks)} chunks.")

    # 3. EMBED & STORE
    # NOTE: if you use Ollama, use
    print("Saving to ChromaDB...")
    db = Chroma.from_documents(
        documents=chunks,
        embedding=OllamaEmbeddings(model="nomic-embed-text"),
        persist_directory=CHROMA_PATH
    )
    print(f"Saved to {CHROMA_PATH}. Ingestion complete!")


if __name__ == "__main__":
    ingest_docs()
