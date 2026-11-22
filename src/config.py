"""Configuração compartilhada do projeto.

Centraliza a leitura do .env e a criação dos modelos de embeddings e de LLM.
O provedor é escolhido pela variável MODEL_PROVIDER (openai ou google).
"""

import os
from functools import lru_cache
from pathlib import Path

from dotenv import load_dotenv
from langchain_core.embeddings import Embeddings
from langchain_core.language_models import BaseChatModel

load_dotenv()

# Banco de dados / ingestão
PDF_PATH = os.getenv("PDF_PATH", "document.pdf")
DATABASE_URL = os.getenv(
    "DATABASE_URL",
    "postgresql+psycopg://postgres:postgres@localhost:5432/rag",
)
COLLECTION_NAME = os.getenv("PG_VECTOR_COLLECTION_NAME", "document_chunks")

# Provedor: "openai" ou "google"
MODEL_PROVIDER = os.getenv("MODEL_PROVIDER", "openai").strip().lower()
SUPPORTED_PROVIDERS = ("openai", "google")

# Modelos OpenAI
OPENAI_EMBEDDING_MODEL = os.getenv(
    "OPENAI_EMBEDDING_MODEL", "text-embedding-3-small"
)
OPENAI_LLM_MODEL = os.getenv("OPENAI_LLM_MODEL", "gpt-4o-mini")

# Modelos Google Gemini
GOOGLE_EMBEDDING_MODEL = os.getenv(
    "GOOGLE_EMBEDDING_MODEL", "gemini-embedding-001"
)
GOOGLE_LLM_MODEL = os.getenv("GOOGLE_LLM_MODEL", "gemini-3.5-flash-lite")

FALLBACK_ANSWER = (
    "Não tenho informações necessárias para responder sua pergunta."
)


def _selected_provider() -> str:
    if MODEL_PROVIDER not in SUPPORTED_PROVIDERS:
        raise RuntimeError(
            f"MODEL_PROVIDER inválido: {MODEL_PROVIDER!r}. "
            "Use 'openai' ou 'google' no arquivo .env."
        )
    return MODEL_PROVIDER


def _require_api_key(var_name: str) -> str:
    value = os.getenv(var_name, "").strip()
    if not value:
        raise RuntimeError(
            f"Variável {var_name} não configurada. "
            "Copie .env.example para .env e preencha a chave do provedor escolhido."
        )
    return value


def get_database_url() -> str:
    """Garante o driver psycopg 3, exigido pelo langchain-postgres."""
    if DATABASE_URL.startswith("postgresql://"):
        return DATABASE_URL.replace(
            "postgresql://", "postgresql+psycopg://", 1
        )
    if DATABASE_URL.startswith("postgres://"):
        return DATABASE_URL.replace(
            "postgres://", "postgresql+psycopg://", 1
        )
    return DATABASE_URL


def get_embeddings() -> Embeddings:
    provider = _selected_provider()

    if provider == "openai":
        from langchain_openai import OpenAIEmbeddings

        return OpenAIEmbeddings(
            model=OPENAI_EMBEDDING_MODEL,
            api_key=_require_api_key("OPENAI_API_KEY"),
        )

    from langchain_google_genai import GoogleGenerativeAIEmbeddings

    return GoogleGenerativeAIEmbeddings(
        model=GOOGLE_EMBEDDING_MODEL,
        google_api_key=_require_api_key("GOOGLE_API_KEY"),
    )


def get_llm() -> BaseChatModel:
    provider = _selected_provider()

    if provider == "openai":
        from langchain_openai import ChatOpenAI

        return ChatOpenAI(
            model=OPENAI_LLM_MODEL,
            temperature=0,
            max_retries=2,
            api_key=_require_api_key("OPENAI_API_KEY"),
        )

    from langchain_google_genai import ChatGoogleGenerativeAI

    return ChatGoogleGenerativeAI(
        model=GOOGLE_LLM_MODEL,
        temperature=0,
        max_retries=2,
        google_api_key=_require_api_key("GOOGLE_API_KEY"),
    )


@lru_cache(maxsize=1)
def get_document_header() -> str:
    """Primeira linha do PDF (cabeçalho da tabela), usada como legenda.

    Os chunks recuperados contêm apenas linhas da tabela. Repetir o cabeçalho
    no início do contexto permite à LLM interpretar corretamente colunas como
    faturamento e ano de fundação, sem inventar informação.
    """
    path = Path(PDF_PATH)
    if not path.is_file():
        return ""

    from langchain_community.document_loaders import PyPDFLoader

    pages = PyPDFLoader(str(path), extraction_mode="layout").load()
    if not pages:
        return ""

    for line in pages[0].page_content.splitlines():
        line = " ".join(line.split())
        if line:
            return line
    return ""


def get_vector_store(embeddings: Embeddings | None = None):
    """Cria a conexão com a collection no PostgreSQL/pgVector."""
    from langchain_postgres import PGVector

    return PGVector(
        embeddings=embeddings or get_embeddings(),
        collection_name=COLLECTION_NAME,
        connection=get_database_url(),
        use_jsonb=True,
    )
