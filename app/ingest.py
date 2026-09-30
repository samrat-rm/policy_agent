"""Load policy markdown, chunk it, embed it, persist to Chroma.

Run once before using the agent:
    python -m app.ingest
"""

from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_ollama import OllamaEmbeddings
from langchain_text_splitters import MarkdownHeaderTextSplitter, RecursiveCharacterTextSplitter

from app.config import (
    CHROMA_DIR,
    CHUNK_OVERLAP,
    CHUNK_SIZE,
    COLLECTION_NAME,
    EMBEDDING_MODEL,
    OLLAMA_BASE_URL,
    POLICY_DIR,
)


def load_documents() -> list[Document]:
    """Read every markdown file in the policy directory."""
    docs = []
    for path in sorted(POLICY_DIR.glob("*.md")):
        docs.append(
            Document(
                page_content=path.read_text(encoding="utf-8"),
                metadata={"source": path.name},
            )
        )
    if not docs:
        raise RuntimeError(f"No .md files found in {POLICY_DIR}")
    return docs


def chunk_documents(docs: list[Document]) -> list[Document]:
    """Split on markdown headers first, then by size.

    Header-aware splitting matters here: policy rules live under headings, and
    a chunk that loses its heading loses the context that makes it answerable.
    Splitting blindly on character count is the single most common cause of
    bad retrieval.
    """
    header_splitter = MarkdownHeaderTextSplitter(
        headers_to_split_on=[("#", "doc_title"), ("##", "section")],
        strip_headers=False,
    )
    size_splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP
    )

    chunks: list[Document] = []
    for doc in docs:
        for header_chunk in header_splitter.split_text(doc.page_content):
            header_chunk.metadata.update(doc.metadata)
            chunks.extend(size_splitter.split_documents([header_chunk]))
    return chunks


def build_vectorstore() -> Chroma:
    docs = load_documents()
    chunks = chunk_documents(docs)
    print(f"Loaded {len(docs)} documents, produced {len(chunks)} chunks.")

    store = Chroma.from_documents(
        documents=chunks,
        embedding=OllamaEmbeddings(model=EMBEDDING_MODEL, base_url=OLLAMA_BASE_URL),
        collection_name=COLLECTION_NAME,
        persist_directory=CHROMA_DIR,
    )
    print(f"Persisted to {CHROMA_DIR}")
    return store


def get_vectorstore() -> Chroma:
    """Open the existing store. Used by the retriever at query time."""
    return Chroma(
        collection_name=COLLECTION_NAME,
        embedding_function=OllamaEmbeddings(model=EMBEDDING_MODEL, base_url=OLLAMA_BASE_URL),
        persist_directory=CHROMA_DIR,
    )


if __name__ == "__main__":
    build_vectorstore()