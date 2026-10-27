# -*- coding: utf-8 -*-
"""
Ask questions about your own Markdown notes (Retrieval-Augmented Generation).

How it works:
  1. Load every .md file in the data/ folder.
  2. Split them into ~2,000-character chunks (200-character overlap so ideas aren't cut mid-thought).
  3. Turn each chunk into an embedding (OpenAI text-embedding-3-small) and store it in a local
     Chroma vector database.
  4. For a question, retrieve the 3 most relevant chunks and ask gpt-4.1-mini to answer
     ONLY from them, citing the source files.

Usage:
    python rag.py "your question" --rebuild   # first run, or after changing the notes
    python rag.py "your question"             # later runs reuse the saved database

Requires the OPENAI_API_KEY environment variable.

Author: Kaustav Palit
"""

from pathlib import Path
from langchain_community.document_loaders import DirectoryLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
import argparse
import shutil
from langchain_core.prompts import ChatPromptTemplate


# Project folder, and the folder holding the Markdown notes to search
BASE_PATH = Path(__file__).parent
DATA_PATH = BASE_PATH / "data"


def load_documents():
    loader = DirectoryLoader(
        path=str(DATA_PATH),
        glob="*.md",
        loader_cls=TextLoader,
        loader_kwargs={"encoding": "utf-8"},
        show_progress=True
    )

    documents = loader.load()
    return documents


def split_documents(documents):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=2000,
        chunk_overlap=200,
        length_function=len,
        add_start_index=True,
    )

    chunks = text_splitter.split_documents(documents)
    return chunks

# Folder in which the local vector database will be saved
CHROMA_PATH = BASE_PATH / "chroma_2000"


def create_vector_store(chunks):
    # Delete the old vector database before rebuilding it
    if CHROMA_PATH.exists():
        shutil.rmtree(CHROMA_PATH)

    embeddings = OpenAIEmbeddings(
        model="text-embedding-3-small"
    )

    db = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=str(CHROMA_PATH)
    )

    return db

PROMPT_TEMPLATE = """
Answer the question based only on the following context.

Context:
{context}

---

Question: {query}

If the context does not contain enough information, say so clearly.
"""

def load_vector_store():
    embeddings = OpenAIEmbeddings(
        model="text-embedding-3-small"
    )

    return Chroma(
        persist_directory=str(CHROMA_PATH),
        embedding_function=embeddings
    )

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "query_text",
        type=str,
        help="The question to search in the RAG database."
    )
    parser.add_argument(
        "--rebuild",
        action="store_true",
        help="Rebuild the Chroma database from the Markdown files first."
    )
    args = parser.parse_args()

    # Build the database only when requested
    if args.rebuild:
        documents = load_documents()
        chunks = split_documents(documents)

        print(f"Loaded {len(documents)} Markdown files.")
        print(f"Created {len(chunks)} chunks.")

        db = create_vector_store(chunks)
        print(f"Vector database created at: {CHROMA_PATH}")
    else:
        if not CHROMA_PATH.exists():
            print("No vector database found. Run again with --rebuild first.")
            return
        db = load_vector_store()

    # Search the existing database using the question you supplied
    results = db.similarity_search_with_relevance_scores(
        args.query_text,
        k=3
    )

    if len(results) == 0:
        print("Unable to find matching results.")
        return

    context_text = "\n\n---\n\n".join(
        document.page_content for document, _score in results
    )

    prompt_template = ChatPromptTemplate.from_template(PROMPT_TEMPLATE)

    prompt = prompt_template.format(
        context=context_text,
        query=args.query_text
    )


    for i, (document, score) in enumerate(results, start=1):
        print(f"\nResult {i} | Relevance score: {score:.3f}")
        print(f"Source: {document.metadata['source']}")
        print(document.page_content)
        print("-" * 70)

    model = ChatOpenAI(
        model="gpt-4.1-mini",
        temperature=0
    )

    response = model.invoke(prompt)
    response_text = response.content

    sources = list(dict.fromkeys(
        document.metadata.get("source", "Unknown source")
        for document, _score in results
    ))

    formatted_response = (
        f"\nResponse:\n{response_text}\n\n"
        f"Sources:\n" + "\n".join(f"- {source}" for source in sources)
    )

    print(formatted_response)


if __name__ == "__main__":
    main()
