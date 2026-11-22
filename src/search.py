"""Busca semântica no pgVector e resposta com a LLM.

A função search_prompt() monta a chain usada pelo chat.py. Cada pergunta é
vetorizada, os 10 trechos mais relevantes são recuperados e enviados à LLM
junto com o prompt que restringe a resposta ao contexto.
"""

from langchain_core.runnables import Runnable, RunnableLambda

from config import (
    FALLBACK_ANSWER,
    get_document_header,
    get_llm,
    get_vector_store,
)

PROMPT_TEMPLATE = """
CONTEXTO:
{contexto}

REGRAS:
- Responda somente com base no CONTEXTO.
- Se a informação não estiver explicitamente no CONTEXTO, responda:
  "Não tenho informações necessárias para responder sua pergunta."
- Nunca invente ou use conhecimento externo.
- Nunca produza opiniões ou interpretações além do que está escrito.

EXEMPLOS DE PERGUNTAS FORA DO CONTEXTO:
Pergunta: "Qual é a capital da França?"
Resposta: "Não tenho informações necessárias para responder sua pergunta."

Pergunta: "Quantos clientes temos em 2024?"
Resposta: "Não tenho informações necessárias para responder sua pergunta."

Pergunta: "Você acha isso bom ou ruim?"
Resposta: "Não tenho informações necessárias para responder sua pergunta."

PERGUNTA DO USUÁRIO:
{pergunta}

RESPONDA A "PERGUNTA DO USUÁRIO"
"""

K = 10
CONTEXT_SEPARATOR = "\n\n---\n\n"


def _friendly_error(exc: Exception) -> str | None:
    """Traduz erros comuns do banco em orientação para o usuário."""
    message = str(exc).lower()

    if "dimension" in message:
        return (
            "A collection foi criada com um modelo de embeddings de outra "
            "dimensão. Rode `python src/ingest.py` para recriá-la com o "
            "provedor/modelo atual."
        )
    if "does not exist" in message or "undefinedtable" in message:
        return (
            "Collection não encontrada no banco. "
            "Rode `python src/ingest.py` para ingerir o PDF."
        )
    return None


def _format_context(documents: list) -> str:
    parts = []
    header = get_document_header()
    if header:
        parts.append(header)
    parts.extend(doc.page_content for doc in documents)
    return CONTEXT_SEPARATOR.join(parts)


def build_chain() -> Runnable:
    """Cria a chain pergunta -> busca -> prompt -> LLM -> resposta."""
    vector_store = get_vector_store()
    llm = get_llm()

    def answer(question: str) -> str:
        question = question.strip()
        if not question:
            return "Digite uma pergunta."

        try:
            results = vector_store.similarity_search_with_score(question, k=K)
        except Exception as exc:
            friendly = _friendly_error(exc)
            if friendly:
                raise RuntimeError(friendly) from None
            raise

        if not results:
            return FALLBACK_ANSWER

        documents = [doc for doc, _score in results]
        prompt = PROMPT_TEMPLATE.format(
            contexto=_format_context(documents),
            pergunta=question,
        )
        response = llm.invoke(prompt)
        return str(response.content).strip()

    return RunnableLambda(answer)


def search_prompt(question: str | None = None):
    """Retorna a chain do chat ou responde uma pergunta isolada.

    Sem argumento, devolve a chain (contrato usado pelo chat.py).
    Com uma pergunta, devolve a resposta em texto.
    """
    try:
        chain = build_chain()
    except Exception as exc:
        print(f"Erro ao inicializar a busca: {exc}")
        return None

    if question is None:
        return chain

    return chain.invoke(question)


if __name__ == "__main__":
    import sys

    pergunta = " ".join(sys.argv[1:]).strip()
    if not pergunta:
        pergunta = input("PERGUNTA: ").strip()

    try:
        resposta = search_prompt(pergunta)
    except Exception as exc:
        print(f"Erro: {exc}")
        raise SystemExit(1) from None

    if resposta is None:
        raise SystemExit(1)

    print(f"RESPOSTA: {resposta}")
