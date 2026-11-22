"""Ingestão do PDF no PostgreSQL com pgVector.

Fluxo: PyPDFLoader (modo layout, preserva a tabela) -> chunks de 1000
caracteres com overlap de 150 -> embeddings -> collection no pgVector.
"""

import re
import time
from pathlib import Path

from langchain_community.document_loaders import PyPDFLoader
from langchain_postgres import PGVector
from langchain_text_splitters import RecursiveCharacterTextSplitter

from config import (
    COLLECTION_NAME,
    PDF_PATH,
    get_database_url,
    get_embeddings,
)

CHUNK_SIZE = 1000
CHUNK_OVERLAP = 150

# Limite de segurança para o free tier do Gemini, que contabiliza cada texto
# enviado em batch_embed_contents como uma requisição por minuto.
EMBED_BATCH_SIZE = 90
MAX_RETRIES = 5


def _is_rate_limit_error(exc: Exception) -> bool:
    message = str(exc).lower()
    return (
        "429" in message
        or "quota" in message
        or "rate limit" in message
        or "resource_exhausted" in message
    )


def _is_daily_quota_error(exc: Exception) -> bool:
    message = str(exc).lower()
    return "per day" in message or "perday" in message or "daily" in message


def _retry_delay_seconds(exc: Exception) -> float:
    match = re.search(r"retry in (\d+(?:\.\d+)?)s", str(exc), re.IGNORECASE)
    if match:
        return float(match.group(1)) + 2
    return 60.0


def _add_batch_with_retry(store: PGVector, batch: list) -> None:
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            store.add_documents(batch)
            return
        except Exception as exc:
            if (
                not _is_rate_limit_error(exc)
                or _is_daily_quota_error(exc)
                or attempt == MAX_RETRIES
            ):
                raise
            delay = _retry_delay_seconds(exc)
            print(
                "Limite de requisições do provedor atingido. "
                f"Aguardando {delay:.0f}s para tentar novamente "
                f"({attempt}/{MAX_RETRIES})..."
            )
            time.sleep(delay)


def ingest_pdf() -> None:
    pdf_path = Path(PDF_PATH)

    if not pdf_path.is_file():
        raise RuntimeError(
            f"PDF não encontrado em {pdf_path.resolve()}. "
            "Confira a variável PDF_PATH no .env."
        )

    print(f"Carregando PDF: {pdf_path}")
    loader = PyPDFLoader(str(pdf_path), extraction_mode="layout")
    pages = loader.load()
    print(f"Páginas carregadas: {len(pages)}")

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
    )
    chunks = splitter.split_documents(pages)
    print(
        f"Chunks gerados: {len(chunks)} "
        f"(chunk_size={CHUNK_SIZE}, overlap={CHUNK_OVERLAP})"
    )

    embeddings = get_embeddings()
    # Valida chave/modelo com uma única chamada antes de apagar a collection.
    # Como pre_delete_collection=True remove os dados na construção do PGVector,
    # uma falha de configuração aqui deixaria o banco vazio.
    print("Validando acesso ao provedor de embeddings...")
    embeddings.embed_query("teste")

    print("Gerando embeddings e gravando no PostgreSQL/pgVector...")

    store = PGVector(
        embeddings=embeddings,
        collection_name=COLLECTION_NAME,
        connection=get_database_url(),
        use_jsonb=True,
        pre_delete_collection=True,
    )

    for start in range(0, len(chunks), EMBED_BATCH_SIZE):
        batch = chunks[start : start + EMBED_BATCH_SIZE]
        _add_batch_with_retry(store, batch)
        stored = min(start + len(batch), len(chunks))
        print(f"Gravados {stored}/{len(chunks)} chunks.")

    print(
        f"Ingestão concluída: {len(chunks)} chunks na collection "
        f"'{COLLECTION_NAME}'."
    )


def main() -> None:
    try:
        ingest_pdf()
    except Exception as exc:
        print(f"Falha na ingestão: {exc}")
        print(
            "Verifique se o banco está no ar (docker compose up -d), "
            "a DATABASE_URL e a chave do provedor no .env."
        )
        raise SystemExit(1) from None


if __name__ == "__main__":
    main()
