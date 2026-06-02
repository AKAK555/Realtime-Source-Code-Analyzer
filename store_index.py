from src.helper import load_repo, repo_ingestion, text_splitter, load_embedding
from dotenv import load_dotenv
from langchain.vectorstores import Chroma
import os

load_dotenv()

GROQ_API_KEY = os.environ.get("GROQ_API_KEY")
os.environ["GROQ_API_KEY"] = GROQ_API_KEY

documents = load_repo("repo/")
text_chunks = text_splitter(documents)
embeddings = load_embedding()

# Create ChromaDB vector store
vectordb = Chroma.from_documents(
    documents=text_chunks,
    embedding=embeddings,
    persist_directory="./db"
)
vectordb.persist()