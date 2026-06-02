import os
import shutil

from git import Repo

from langchain_community.document_loaders import TextLoader

from langchain_text_splitters import RecursiveCharacterTextSplitter

from langchain_huggingface import HuggingFaceEmbeddings

# Clone any repository
def repo_ingestion(repo_url):

    repo_name = repo_url.rstrip("/").split("/")[-1]

    repo_path = os.path.join("repos", repo_name)

    os.makedirs("repos", exist_ok=True)

    if not os.path.exists(repo_path):

        print(f"Cloning {repo_name}...")

        Repo.clone_from(repo_url, repo_path)

    else:

        print(f"{repo_name} already exists.")

    return repo_path


# Load all relevant files from the repository as documents
def load_repo(repo_path):

    allowed_extensions = (
    ".py",
    ".md",
    ".txt"
    )

    documents = []

    for root, dirs, files in os.walk(repo_path):

        # Ignore unnecessary folders
        dirs[:] = [
            d for d in dirs
            if d not in {
                ".git",
                "__pycache__",
                "node_modules",
                "venv",
                ".venv",
                "dist",
                "build",
                ".idea",
                ".vscode",
                "docs"
            }
        ]

        for file in files:

            if file.lower().endswith(allowed_extensions):

                file_path = os.path.join(root, file)

                try:
                    loader = TextLoader(
                        file_path,
                        encoding="utf-8"
                    )

                    documents.extend(loader.load())

                except Exception as e:
                    print(f"Skipped {file_path}: {e}")

    return documents

# Create chunks
def text_splitter(documents):

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=150,
        separators=[
            "\nclass ",
            "\ndef ",
            "\nfunction ",
            "\n\n",
            "\n",
            " ",
            ""
        ]
    )

    chunks = splitter.split_documents(documents)

    return chunks

# Load Embeddings
def load_embedding():
    embeddings = HuggingFaceEmbeddings(
        model_name="BAAI/bge-small-en-v1.5"
    )
    return embeddings

from langchain_chroma import Chroma

def build_index(repo_url):

    repo_path = repo_ingestion(repo_url)
    print("Loading repository...")
    documents = load_repo(repo_path)
    print(f"Loaded {len(documents)} files")

    print("Chunking...")
    chunks = text_splitter(documents)

    print(f"Created {len(chunks)} chunks")

    print("Generating embeddings...")
    embeddings = load_embedding()
    
    repo_name = os.path.basename(repo_path)

    db_path = f"./db/{repo_name}"

    if os.path.exists(db_path):
        shutil.rmtree(db_path)
    
    Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=db_path
    )
    return db_path

