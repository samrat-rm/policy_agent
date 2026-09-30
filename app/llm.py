"""Provider factory.

Every LLM and embedding object in the project comes from here. Nothing else
imports a provider SDK directly, so switching backends is an env var rather
than an edit across several files.

    LLM_PROVIDER=openai   EMBEDDING_PROVIDER=openai   # fast, costs money
    LLM_PROVIDER=ollama   EMBEDDING_PROVIDER=ollama   # free, slow, local

Mixing is allowed and often sensible: OpenAI for chat, Ollama for embeddings
keeps iteration fast while spending nothing on the part that runs once.
"""

from app.config import (
    EMBEDDING_MODEL,
    EMBEDDING_PROVIDER,
    LLM_MODEL,
    LLM_PROVIDER,
    OLLAMA_BASE_URL,
)


def get_chat_model(temperature: float = 0):
    """Build the chat model for the configured provider."""
    if LLM_PROVIDER == "openai":
        from langchain_openai import ChatOpenAI

        return ChatOpenAI(model=LLM_MODEL, temperature=temperature)

    if LLM_PROVIDER == "ollama":
        from langchain_ollama import ChatOllama

        return ChatOllama(
            model=LLM_MODEL, temperature=temperature, base_url=OLLAMA_BASE_URL
        )

    raise ValueError(
        f"Unknown LLM_PROVIDER '{LLM_PROVIDER}'. Use 'openai' or 'ollama'."
    )


def get_embeddings():
    """Build the embedding model for the configured provider.

    Changing this invalidates your vector store. Embedding models produce
    vectors of different dimensions (text-embedding-3-small is 1536,
    nomic-embed-text is 768), and a store built with one cannot be queried
    with the other. Delete chroma_db/ and re-run `python -m app.ingest`
    whenever you change this setting.
    """
    if EMBEDDING_PROVIDER == "openai":
        from langchain_openai import OpenAIEmbeddings

        return OpenAIEmbeddings(model=EMBEDDING_MODEL)

    if EMBEDDING_PROVIDER == "ollama":
        from langchain_ollama import OllamaEmbeddings

        return OllamaEmbeddings(model=EMBEDDING_MODEL, base_url=OLLAMA_BASE_URL)

    raise ValueError(
        f"Unknown EMBEDDING_PROVIDER '{EMBEDDING_PROVIDER}'. "
        "Use 'openai' or 'ollama'."
    )


def describe() -> str:
    """One-line summary, used in eval reports so runs are comparable."""
    return f"{LLM_PROVIDER}:{LLM_MODEL} / emb {EMBEDDING_PROVIDER}:{EMBEDDING_MODEL}"