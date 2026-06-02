from src.helper import (
    load_repo,
    repo_ingestion,
    text_splitter,
    load_embedding
)

from dotenv import load_dotenv
from langchain_chroma import Chroma
import os

load_dotenv()

# Optional if using Groq later
groq_api_key = os.getenv("GROQ_API_KEY")

# Clone repository
repo_url = input("Enter GitHub Repository URL: ")

repo_path = repo_ingestion(repo_url)

# Load documents
documents = load_repo(repo_path)

print(f"Loaded {len(documents)} documents")

# Create chunks
text_chunks = text_splitter(documents)

print(f"Created {len(text_chunks)} chunks")

# Load embeddings
embeddings = load_embedding()

# Create vector store
vectordb = Chroma.from_documents(
    documents=text_chunks,
    embedding=embeddings,
    persist_directory="./db"
)

print("Vector database created successfully!")