"""CLI de perguntas e respostas sobre o PDF ingerido."""

from search import search_prompt

EXIT_COMMANDS = {"sair", "exit", "quit", "q"}


def main() -> None:
    chain = search_prompt()

    if not chain:
        print("Não foi possível iniciar o chat. Verifique os erros de inicialização.")
        return

    print("Chat de busca semântica sobre o PDF.")
    print("Digite sua pergunta ou 'sair' para encerrar.")

    while True:
        try:
            question = input("\nPERGUNTA: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nEncerrando.")
            break

        if not question:
            continue

        if question.lower() in EXIT_COMMANDS:
            print("Encerrando.")
            break

        try:
            answer = chain.invoke(question)
        except Exception as exc:
            print(f"RESPOSTA: Erro ao consultar: {exc}")
            print("\n---")
            continue

        print(f"RESPOSTA: {answer}")
        print("\n---")


if __name__ == "__main__":
    main()
