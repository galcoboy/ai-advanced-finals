"""
Assignment 3 — RAG with a Word Document
Loads sample_document.docx, chunks, embeds in ChromaDB, answers 5 questions with Gemini.
Prints LLM answers and retrieved context for each question.
"""

import os
from pathlib import Path

from dotenv import load_dotenv
from langchain_community.document_loaders import Docx2txtLoader
from langchain_community.embeddings import SentenceTransformerEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_text_splitters import RecursiveCharacterTextSplitter

load_dotenv(Path(__file__).resolve().parent.parent / "assignment4_restaurant" / ".env")
load_dotenv()

DOC_PATH = Path(__file__).parent / "sample_document.docx"
CHROMA_DIR = Path(__file__).parent / "chroma_docx_db"

# ── STEP 1: Load Word document ───────────────────────────────────────────────
loader = Docx2txtLoader(str(DOC_PATH))
docs = loader.load()
print(f"Loaded {len(docs)} document(s), {len(docs[0].page_content)} characters")

# ── STEP 2: Chunk ────────────────────────────────────────────────────────────
splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=60)
chunks = splitter.split_documents(docs)
print(f"Split into {len(chunks)} chunks\n")

# ── STEP 3: Embed and store in ChromaDB (free local embeddings) ──────────────
embeddings = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")
vectorstore = Chroma.from_documents(
    documents=chunks,
    embedding=embeddings,
    persist_directory=str(CHROMA_DIR),
)
print("Vector store created and persisted.\n")

retriever = vectorstore.as_retriever(search_kwargs={"k": 3})

api_key = os.getenv("GOOGLE_API_KEY")
if not api_key:
    raise SystemExit("GOOGLE_API_KEY required in .env for Gemini answers")

llm = ChatGoogleGenerativeAI(model="gemini-flash-latest", temperature=0, google_api_key=api_key)

prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "Answer the question using ONLY the context below. "
            "If the context does not contain the answer, say you are not sure.",
        ),
        ("human", "Question: {question}\n\nContext:\n{context}"),
    ]
)
chain = prompt | llm | StrOutputParser()

QUESTIONS = [
    "What problem does RAG solve that a plain LLM cannot?",
    "What are the main steps of the ingestion pipeline?",
    "What chunk size and overlap settings are recommended, and why does overlap matter?",
    "Who formalized the RAG architecture and in which year?",
    "What are two limitations of RAG mentioned in the document?",
]

print("=" * 70)
for question in QUESTIONS:
    source_docs = retriever.invoke(question)
    context = "\n\n---\n\n".join(d.page_content for d in source_docs)
    answer = chain.invoke({"question": question, "context": context})

    print(f"\nQuestion: {question}")
    print(f"Answer: {answer}")
    print("\nRetrieved context:")
    for i, src in enumerate(source_docs, 1):
        preview = src.page_content.replace("\n", " ")[:220]
        print(f"  [{i}] ...{preview}...")
    print("-" * 70)
