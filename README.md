# Ask Your Notes: A Simple RAG Question-Answering Tool

A command-line tool that answers questions from **your own Markdown documents**, citing which files the answer came from. It uses Retrieval-Augmented Generation (RAG): instead of relying on what the AI model already "knows", it finds the relevant passages in your notes and answers only from those.

```
$ python rag.py "When does the Northwind Library close?" --rebuild

Loaded 1 Markdown files.
Created 1 chunks.
...
Response:
The Northwind Library closes at 8 pm on weekdays and is closed on public holidays.

Sources:
- data/example_notes.md
```

## How it works

```
data/*.md ──► split into chunks ──► embeddings ──► Chroma vector DB (local)
                                                         │
your question ──► embedding ──► find 3 closest chunks ◄──┘
                                       │
                                       ▼
              gpt-4.1-mini answers ONLY from those chunks + lists sources
```

| Step | Choice | Why |
|---|---|---|
| Chunking | 2,000 characters, 200 overlap | Big enough to keep an idea together; the overlap avoids cutting a point in half |
| Embeddings | OpenAI `text-embedding-3-small` | Cheap and accurate for semantic search |
| Vector store | Chroma, saved to disk | No server needed; build once, reuse on later runs |
| Answering | `gpt-4.1-mini`, temperature 0 | Consistent answers; told to say so when the notes don't contain the answer |

Grounding the answer in retrieved text, and showing the sources, reduces made-up answers and lets you check every claim.

## Setup

```bash
pip install -r requirements.txt
```

Set your OpenAI API key as an environment variable (see `.env.example`):

```bash
# Windows (PowerShell)
$env:OPENAI_API_KEY="sk-..."
# macOS / Linux
export OPENAI_API_KEY="sk-..."
```

## Usage

1. Put your Markdown files in the `data/` folder (an example file is included).
2. Build the database and ask a question:
   ```bash
   python rag.py "your question" --rebuild
   ```
3. Ask more questions without rebuilding:
   ```bash
   python rag.py "another question"
   ```

Re-run with `--rebuild` whenever you add or change files in `data/`.

## Ideas for extension

- Support PDFs and Word files as well as Markdown
- A relevance-score cut-off, so weak matches are not used as context
- A simple web interface (e.g. Streamlit) instead of the command line
- Swap in an open-source embedding model and LLM to run fully offline

## Tech

Python · LangChain · Chroma · OpenAI API
