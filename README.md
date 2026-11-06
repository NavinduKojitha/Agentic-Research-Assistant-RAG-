# Agentic Research Assistant (RAG)

A Gradio web app that answers questions about any PDF you upload, using retrieval-augmented generation (RAG) with LangChain, FAISS and OpenAI.

Upload a research paper or report, ask a question in plain English, and the assistant retrieves the most relevant passages from the document and uses an LLM to write the answer.

## How it works

```
PDF upload ──> PyPDFLoader ──> OpenAI embeddings ──> FAISS vector index
                                                          │
Question ──> similarity search over the index ────────────┘
                     │
                     └──> retrieved passages + question ──> ChatOpenAI ──> answer
```

1. **Load** – the PDF is split into pages with LangChain's `PyPDFLoader`.
2. **Embed and index** – each page is embedded with `OpenAIEmbeddings` and stored in an in-memory FAISS index.
3. **Retrieve** – for each question, the most similar pages are retrieved from FAISS.
4. **Generate** – a `RetrievalQA` chain passes the retrieved context and the question to `ChatOpenAI` (temperature 0.1) to produce a grounded answer.

## Features

- Upload any PDF and query it in natural language
- OpenAI API key entered in a password field at runtime; it is never written to disk or committed
- Preset research questions (AI, quantum computing, Web3, market trends) for a quick demo
- Step-by-step progress log showing each stage of the pipeline
- Clean monochrome Gradio interface

## Tech stack

Python · LangChain · FAISS · OpenAI API (embeddings + chat) · Gradio · pypdf

## Getting started

**Requirements:** Python 3.10 or 3.11 (the pinned `faiss-cpu` version has no wheels for newer Python) and an OpenAI API key.

```bash
git clone https://github.com/NavinduKojitha/Agentic-Research-Assistant-RAG-.git
cd Agentic-Research-Assistant-RAG-

python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate

pip install -r requirements.txt
python app.py
```

Open the local URL Gradio prints (usually http://127.0.0.1:7860), then:

1. Upload a PDF.
2. Paste your OpenAI API key and click **Load Agent**.
3. Type a question, or click one of the preset topics.

## Project structure

```
├── app.py                 # Gradio UI and event handlers
├── rag_agent.py           # RAG pipeline: load PDF, build FAISS index, RetrievalQA chain
├── requirements.txt
└── Agentic_Research_Assistant_Project_Report.pdf   # project report
```

## Possible improvements

- Split pages into smaller overlapping chunks for more precise retrieval
- Show the source page numbers alongside each answer
- Persist the FAISS index so large PDFs are not re-embedded on every load
- Add a tool-using agent (e.g. web search) for questions the document cannot answer

## Author

**Navindu Balasooriya** – AI Engineer
[LinkedIn](https://www.linkedin.com/in/navindu-balasooriya-b3a3b71b9/) · navindukb@gmail.com
