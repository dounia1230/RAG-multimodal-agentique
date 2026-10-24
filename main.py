"""Run with: uv run streamlit run main.py"""

import hashlib
from pathlib import Path

import pymupdf
import streamlit as st

from rag import DEFAULT_PDF, ROOT, ask, display_filename, index_pdf, is_indexed, pdf_id


def pdf_page_image(pdf_path, page_number):
    """Show the original layout so equations remain readable."""
    with pymupdf.open(pdf_path) as pdf:
        return pdf[page_number - 1].get_pixmap(matrix=pymupdf.Matrix(1.5, 1.5), alpha=False).tobytes("png")


def main():
    st.set_page_config(page_title="Attention Is All You Need — RAG", page_icon="📄")
    st.title("📄 Ask Attention Is All You Need")
    st.caption("A simple local RAG: PDF → text / images / tables → pgvector → Qwen")

    with st.sidebar:
        st.header("1. Select a PDF")
        uploaded = st.file_uploader("Use your own PDF, or keep the included paper", type="pdf")
        pdf_path = DEFAULT_PDF
        if uploaded:
            # Use a content hash so uploads with the same name cannot overwrite each other.
            name = hashlib.sha256(uploaded.getvalue()).hexdigest()[:16]
            folder = ROOT / "artifacts" / "uploads" / name
            folder.mkdir(parents=True, exist_ok=True)
            pdf_path = folder / Path(uploaded.name).name
            if not pdf_path.exists():
                pdf_path.write_bytes(uploaded.getvalue())
        st.write(f"Selected: **{display_filename(pdf_path.name)}**")
        if not pdf_path.exists():
            st.error("The default paper is missing. Upload a PDF to continue.")
            st.stop()

        if st.button("2. Index this PDF", type="primary"):
            try:
                with st.status("Indexing the PDF...", expanded=True) as status:
                    counts = index_pdf(pdf_path, progress=st.write)
                    status.update(label="PDF indexed", state="complete")
                st.success(f"{counts['text_chunks']} text chunks · {counts['images']} images · "
                           f"{counts['tables']} tables")
            except Exception as error:
                st.error(f"Indexing failed: {error}")
                st.stop()
        st.caption("Index once. The first run loads the text and CLIP embedding models.")

    try:
        ready = is_indexed(pdf_path)
    except Exception:
        st.error("Cannot connect to pgvector. Check Docker and the PostgreSQL settings in .env.")
        st.stop()
    if not ready:
        st.info("Click ‘Index this PDF’ in the sidebar first.")
        st.stop()

    document_id = pdf_id(pdf_path)
    if st.session_state.get("document_id") != document_id:
        st.session_state.document_id = document_id
        st.session_state.result = None

    st.subheader("3. Ask a question")
    st.caption("Try: Explain Figure 1. · Compare the BLEU scores in Table 2. · What is multi-head attention?")
    with st.form("question"):
        question = st.text_input("Your question", value="What is multi-head attention?")
        submitted = st.form_submit_button("Ask")
    if submitted and question.strip():
        st.session_state.result = None
        try:
            with st.spinner("Searching the PDF and generating an answer..."):
                st.session_state.result = ask(question, pdf_path)
        except Exception as error:
            st.error(f"Could not answer: {error}. Check that Ollama and the configured model are available.")

    result = st.session_state.get("result")
    if result:
        st.subheader("Answer")
        st.markdown(result["answer"])
        st.subheader("Sources")
        for source in result["sources"]:
            label = (f"[{source['id']}] {display_filename(source['file'])} "
                     f"— page {source['page']} ({source['kind']})")
            with st.expander(label, expanded=False):
                path = source["image_path"]
                if path and Path(path).exists():
                    st.image(path, width="stretch")
                else:
                    st.image(pdf_page_image(pdf_path, source["page"]), width="stretch")
        with st.expander("Agent search steps"):
            st.json(result["tool_calls"])


if __name__ == "__main__":
    main()
