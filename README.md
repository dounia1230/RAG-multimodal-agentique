# Simple multimodal agentic RAG

Ask questions about **Attention Is All You Need** using your existing Docker
PostgreSQL/pgvector database and local Ollama models. No API key is needed.

## Run it (PowerShell)

From this project folder:

```powershell
uv sync
# Only copy this if you have not already configured .env:
Copy-Item .env.example .env
uv run streamlit run main.py
```

Your container is already running on **localhost:5433**; the defaults in
`.env.example` match `docker-compose.yml`. If needed, start it again with
`docker compose up -d`. Ollama must also be running. The answer model found on this
machine is `Qwen3:8b`. On another machine, install it with:

```powershell
ollama pull Qwen3:8b
```

Open the Streamlit URL, keep the included PDF, click **Index this PDF**, then ask
a question. Indexing computes text and image embeddings. Embedding weights download on first use if not cached.
Reopening the app reuses the database. Reindexing replaces entries for the same
PDF content without accumulating duplicates; other PDFs remain stored separately.

## Read the code in this order

1. **`rag.ipynb`** — short notebook walkthrough. Select this project's `.venv`
   Python kernel in your IDE and run the cells in order.
2. **`rag.py`** — the pipeline, divided into six commented steps.
3. **`main.py`** — the Streamlit upload, indexing button, question form and sources.

There are no custom classes, cloud services or additional database containers.

## What happens?

```text
PDF
 ├─ LlamaIndex reader → text chunks ──────────────┐
 ├─ PyMuPDF → table text + screenshots ───────────┤
 └─ PyMuPDF → figures → PDF captions ───────┤
                                                ↓
                              MiniLM text embeddings (384)
                                                ↓
                                   pgvector text table

Figures → CLIP image embeddings (512) → pgvector image table

Question → initial text search → LangChain agent (Qwen)
                               ├─ search_text
                               ├─ search_images (CLIP + captions)
                               └─ search_tables
                                       ↓
                         Answer + source IDs + source images
```

- **LlamaIndex** reads the PDF, splits text and stores/retrieves nodes through
  `PGVectorStore`.
- **MiniLM** embeds text, table text and PDF figure captions. **CLIP**
  embeds figures and short image-search queries in a shared visual space.
  These are different vector spaces, so they use separate database tables.
- **CLIP** uses `ClipEmbedding` from `llama-index-embeddings-clip` and the
  `openai/CLIP` package installed from GitHub. It does not generate descriptions.
  We index the original PDF captions instead. Qwen receives retrieved text and
  captions; the interface displays the original figures and table screenshots.
- **LangChain** exposes three search functions as tools. Qwen chooses which to
  call and can search again. Expand **Agent search steps** to see the queries.
- Sources include the filename and **PDF page number (starting at 1)**.

Answers use Markdown with LaTeX equations. Each source has a collapsed panel
showing the original PDF page, figure or table, so equations keep their layout.
Upload storage hashes are hidden from filenames. The notebook renders answers
as Markdown too.

The project creates `data_course_text_v1` and `data_course_images_v1` inside the
existing database. Keep `artifacts/` with the database: it contains the images
referenced by stored metadata. Indexing computes all embeddings before replacing
the selected PDF's entries; the final writes to the two tables are not atomic.
If a database write fails, run indexing again.

## Try these questions

- What is multi-head attention?
- Explain the encoder and decoder in Figure 1.
- What does Figure 2 show?
- Compare the Transformer base and big BLEU scores in Table 2.
- According to Table 1, how does self-attention complexity compare with recurrence?
- What optimizer and learning rate schedule were used?

English retrieval works best because the paper and embedding models are English.
You can ask in French; the agent is instructed to search in English and answer
in your question's language. Each question is independent (no chat memory).

## Limits for this classroom example

Table extraction preserves original text and screenshots rather than trying to
rebuild complex merged cells as a DataFrame. It handles this paper's horizontal
table rules and uses PyMuPDF's table detector for other layouts. Borderless
tables and vector-only figures can fall back to a full-page screenshot.
Scanned PDFs need OCR, which this example does not include. CLIP image
search queries must fit its 77-token limit, so keep them short. Local-model answers can
be wrong; check the displayed sources, especially equations and exact numbers.

The slide suggests a machine-maintenance document; this project uses the paper
you requested. You can upload a maintenance PDF through the same interface.

## References

- [Attention Is All You Need](https://arxiv.org/abs/1706.03762)
- [LlamaIndex PostgreSQL integration](https://developers.llamaindex.ai/python/framework/integrations/vector_stores/postgres/)
- [PyMuPDF page extraction and tables](https://pymupdf.readthedocs.io/en/latest/page.html)
- [OpenAI CLIP](https://github.com/openai/CLIP)
- [LangChain agents](https://docs.langchain.com/oss/python/langchain/agents)
- [ChatOllama](https://docs.langchain.com/oss/python/integrations/chat/ollama)
