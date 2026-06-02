from flask import Flask, request, jsonify, render_template
from dotenv import load_dotenv
import os

from langchain_huggingface import HuggingFaceEmbeddings
from langchain_groq import ChatGroq

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from src.helper import (
    load_embedding,
    build_index
)

from langchain_chroma import Chroma
import atexit
import shutil
import os

CURRENT_DB_PATH = None

def cleanup():

    print("Cleaning up...")

    try:
        if os.path.exists("repos"):
            shutil.rmtree("repos", ignore_errors=True)

        if os.path.exists("db"):
            shutil.rmtree("db", ignore_errors=True)

    except Exception as e:
        print(f"Cleanup Error: {e}")


atexit.register(cleanup)


load_dotenv()

app = Flask(__name__)

embeddings = load_embedding()

vectordb = None
retriever = None

#Function to reload DB
def reload_retriever():

    global vectordb
    global retriever
    global CURRENT_DB_PATH

    vectordb = Chroma(
        persist_directory=CURRENT_DB_PATH,
        embedding_function=embeddings
    )
    retriever = vectordb.as_retriever(
    search_type="mmr",
    search_kwargs={
        "k": 8,
        "fetch_k": 30
        }
    )
    print("Collection size:", vectordb._collection.count())


# Groq LLM
llm = ChatGroq(
    model="llama-3.3-70b-versatile",
    temperature=0,
    api_key=os.getenv("GROQ_API_KEY")
)

# Prompt
prompt = ChatPromptTemplate.from_template(
"""
You are an expert GitHub Repository Analyzer.

Rules:

1. Answer ONLY from the supplied repository context.

2. If the answer is not explicitly present in the context, reply exactly:

I could not find that information in the indexed repository.

3. When discussing code, mention the source file(s).

4. For repository overview questions:
   - Summarize based on README.md if available.
   - Mention the main technologies and purpose.

5. Never invent functions, classes, files, modules or behaviour.

Repository Context:
{context}

Question:
{question}

Answer:
"""
)

chain = prompt | llm | StrOutputParser()

@app.route("/")
def index():
    return render_template("index.html")


@app.route("/chatbot", methods=["GET", "POST"])
def gitRepo():

    global CURRENT_DB_PATH

    if request.method == "POST":

        repo_url = request.form["question"]

        CURRENT_DB_PATH = build_index(repo_url)

        reload_retriever()

        return jsonify(
            {
                "response": f"Repository indexed successfully"
            }
        )
    return render_template("index.html")
        
@app.route("/get", methods=["POST"])
def chat():

    import re

    user_input = request.form["msg"]

    if user_input.lower() == "clear":
        return "Please restart the application before clearing the database."

    if retriever is None:
        return "Please index a repository first."

    docs = []

    # ====================================================
    # REPOSITORY OVERVIEW QUESTIONS
    # ====================================================

    overview_keywords = [
        "what is this repository about",
        "what does this repository do",
        "purpose of this repository",
        "repository summary",
        "repository overview",
        "explain this repository",
        "what is this project"
    ]

    if any(
        phrase in user_input.lower()
        for phrase in overview_keywords
    ):

        print("\nREADME SEARCH MODE")

        candidate_docs = vectordb.similarity_search(
            "README.md",
            k=20
        )

        docs = [
            d for d in candidate_docs
            if "readme" in
            d.metadata.get("source", "").lower()
        ]

        print(
            f"README CHUNKS FOUND: {len(docs)}"
        )

    # ====================================================
    # FILE-SPECIFIC QUESTIONS
    # ====================================================

    if len(docs) == 0:

        file_match = re.search(
            r'([\w\-]+\.(py|js|ts|java|cpp|c|go|rs|php|rb|md|txt))',
            user_input,
            re.IGNORECASE
        )

        if file_match:

            filename = file_match.group(1)

            print(
                f"\nFILE QUERY DETECTED: {filename}"
            )

            candidate_docs = vectordb.similarity_search(
                filename,
                k=30
            )

            docs = [
                d for d in candidate_docs
                if filename.lower()
                in d.metadata.get(
                    "source",
                    ""
                ).lower()
            ]

            print(
                f"FOUND {len(docs)} CHUNKS"
            )

    # ====================================================
    # NORMAL SEMANTIC SEARCH
    # ====================================================

    if len(docs) == 0:

        print("\nSEMANTIC SEARCH MODE")

        docs = retriever.invoke(user_input)

    # ====================================================
    # DEBUG
    # ====================================================

    print("=" * 80)
    print("QUESTION:", user_input)
    print("DOCS RETRIEVED:", len(docs))

    for i, doc in enumerate(docs):

        print(f"\nDOCUMENT {i+1}")

        print(
            "SOURCE:",
            doc.metadata.get("source")
        )

        print(doc.page_content[:500])

    print("=" * 80)

    # ====================================================
    # CONTEXT
    # ====================================================

    context = "\n\n".join(
        f"""
FILE:
{doc.metadata.get('source')}

CONTENT:
{doc.page_content}
"""
        for doc in docs
    )

    # ====================================================
    # LLM
    # ====================================================

    answer = chain.invoke(
        {
            "context": context,
            "question": user_input
        }
    )

    return answer

if __name__ == "__main__":
    app.run(
        host="0.0.0.0",
        port=8080,
        debug=True,
        use_reloader=False
    )