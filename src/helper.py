import os
from git import Repo
from langchain.document_loaders.generic import GenericLoader
from langchain.document_loaders.parsers import LanguageParser
from langchain.text_splitter import RecursiveCharacterTextSplitter, Language
from langchain.embeddings.groq import GroqEmbeddings

import os
import shutil
from git import Repo

# Clone any repository
def repo_ingestion(repo_url):

    repo_path = "repo"

    if os.path.exists(repo_path):
        shutil.rmtree(repo_path)

    Repo.clone_from(repo_url, repo_path)

    return repo_path


# Load all relevant files from the repository as documents
from langchain_community.document_loaders import DirectoryLoader, TextLoader


def load_repo(repo_path):

    allowed_extensions = (
        ".py",
        ".java",
        ".js",
        ".ts",
        ".cpp",
        ".c",
        ".h",
        ".hpp",
        ".go",
        ".rs",
        ".php",
        ".rb",
        ".md",
        ".txt",
        ".json",
        ".yaml",
        ".yml",
        ".toml",
        ".ini",
        ".cfg",
        ".html",
        ".css",
        ".sql",
        ".sh",
        ".dockerfile"
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
                ".vscode"
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
from langchain.text_splitter import RecursiveCharacterTextSplitter


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
from langchain_huggingface import HuggingFaceEmbeddings


def load_embedding():

    embeddings = HuggingFaceEmbeddings(
        model_name="BAAI/bge-small-en-v1.5"
    )

    return embeddings