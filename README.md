# Ingestão e Busca Semântica com LangChain e Postgres

## Objetivo

Você deve entregar um software capaz de:

- Ingestão: Ler um arquivo PDF e salvar suas informações em um banco de dados PostgreSQL com extensão pgVector.
- Busca: Permitir que o usuário faça perguntas via linha de comando (CLI) e receba respostas baseadas apenas no conteúdo do PDF.

## Exemplo no CLI

Faça sua pergunta:

```
PERGUNTA: Qual o faturamento da Empresa SuperTechIABrazil?
RESPOSTA: O faturamento foi de 10 milhões de reais.

---

Perguntas fora do contexto:

PERGUNTA: Quantos clientes temos em 2024?
RESPOSTA: Não tenho informações necessárias para responder sua pergunta.
```

## Tecnologias obrigatórias

- Linguagem: Python
- Framework: LangChain
- Banco de dados: PostgreSQL + pgVector
- Execução do banco de dados: Docker & Docker Compose (docker-compose fornecido no repositório de exemplo)

## Pacotes recomendados

- Split: `from langchain_text_splitters import RecursiveCharacterTextSplitter`
- Embeddings (OpenAI): `from langchain_openai import OpenAIEmbeddings`
- Embeddings (Gemini): `from langchain_google_genai import GoogleGenerativeAIEmbeddings`
- PDF: `from langchain_community.document_loaders import PyPDFLoader`
- Ingestão: `from langchain_postgres import PGVector`
- Busca: `similarity_search_with_score(query, k=10)`

## OpenAI

- Crie uma API Key da OpenAI.
- Você vai precisar de um modelo de embeddings e de um modelo de LLM para responder. Consulte a documentação oficial da OpenAI para ver os modelos disponíveis.

## Gemini

- Crie uma API Key da Google.
- Você vai precisar de um modelo de embeddings e de um modelo de LLM para responder. Consulte a documentação oficial do Google para ver os modelos disponíveis.

Os limites de requisições gratuitas dos modelos podem mudar com frequência. Para informações atualizadas, consulte a documentação oficial do Google.

## Escolha dos modelos

Este desafio não fixa modelos. Nomes e versões mudam com frequência e alguns são descontinuados, então faz parte do desafio consultar a documentação oficial do provedor que você escolher, ver quais modelos estão disponíveis no momento e selecionar os que atendem ao objetivo. Para o volume deste desafio, os modelos mais leves e baratos de cada provedor são suficientes.

Atenção: modelos de embedding diferentes geram vetores com dimensões diferentes. A tabela de vetores é criada na primeira ingestão, já com a dimensão do modelo que você escolheu. Se você trocar de modelo de embeddings depois disso, a ingestão passa a falhar por incompatibilidade de dimensão. Nesse caso é responsabilidade sua apagar a collection existente (ou o volume do banco) e refazer a ingestão do zero com o novo modelo.

## Requisitos

### 1. Ingestão do PDF

- O PDF deve ser dividido em chunks de 1000 caracteres com overlap de 150.
- Cada chunk deve ser convertido em embedding.
- Os vetores devem ser armazenados no banco de dados PostgreSQL com pgVector.

### 2. Consulta via CLI

Criar um script Python para simular um chat no terminal.

Passos ao receber uma pergunta:

- Vetorizar a pergunta.
- Buscar os 10 resultados mais relevantes (k=10) no banco vetorial.
- Montar o prompt e chamar a LLM.
- Retornar a resposta ao usuário.

Prompt a ser utilizado:

```
CONTEXTO:
{resultados concatenados do banco de dados}

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
{pergunta do usuário}

RESPONDA A "PERGUNTA DO USUÁRIO"
```

## Estrutura obrigatória do projeto

Faça um fork do repositório para utilizar a estrutura abaixo: https://github.com/devfullcycle/mba-ia-desafio-ingestao-busca

```
├── docker-compose.yml
├── requirements.txt      # Dependências
├── .env.example          # Template das variáveis de ambiente
├── src/
│   ├── ingest.py         # Script de ingestão do PDF
│   ├── search.py         # Script de busca
│   ├── chat.py           # CLI para interação com usuário
├── document.pdf          # PDF para ingestão
└── README.md             # Instruções de execução
```

## VirtualEnv para Python

Crie e ative um ambiente virtual antes de instalar dependências:

```
python3 -m venv venv
source venv/bin/activate
```

## Ordem de execução

1. Subir o banco de dados:

```
docker compose up -d
```

2. Executar ingestão do PDF:

```
python src/ingest.py
```

3. Rodar o chat:

```
python src/chat.py
```

## Entregável

Repositório público no GitHub contendo todo o código-fonte e README com instruções claras de execução do projeto.

---

# Solução implementada

Software de **RAG (Retrieval-Augmented Generation)** que lê o PDF, armazena os
trechos vetorizados no PostgreSQL com pgVector e responde perguntas pelo
terminal usando somente o conteúdo do documento. O `document.pdf` deste
repositório contém uma tabela de empresas com **nome, faturamento e ano de
fundação**.

- Ingestão em chunks de **1000 caracteres com overlap de 150**.
- Embeddings por **OpenAI** ou **Google Gemini**, escolhido no `.env`.
- Busca dos **10 trechos mais relevantes** (`k=10`) para cada pergunta.
- CLI que só responde com base no contexto recuperado.

## Estrutura do projeto

```
├── docker-compose.yml
├── requirements.txt
├── .env.example
├── document.pdf
├── README.md
└── src/
    ├── config.py    # Leitura do .env e criação de embeddings/LLM
    ├── ingest.py    # Ingestão do PDF no pgVector
    ├── search.py    # Busca semântica + prompt + LLM
    └── chat.py      # CLI de interação com o usuário
```

## Pré-requisitos

- Docker e Docker Compose instalados e em execução.
- Python **3.11 ou superior** — os pacotes fixados usam `numpy` 2.3, que exige
  3.11+. O projeto foi validado com **Python 3.11.15**.
- Uma API Key da OpenAI **ou** do Google Gemini.

## Configuração

### 1. Ambiente virtual e dependências

```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

> Se o seu `python3.11` não estiver no PATH, instale-o com `pyenv` ou `uv` e
> crie o venv com o caminho completo. Exemplo com pyenv:
>
> ```bash
> pyenv install 3.11.15
> pyenv local 3.11.15
> python -m venv venv
> ```

### 2. Variáveis de ambiente

```bash
cp .env.example .env
```

Edite o `.env` e escolha o provedor:

| Variável | Descrição | Padrão |
| --- | --- | --- |
| `MODEL_PROVIDER` | `openai` ou `google` | `openai` |
| `OPENAI_API_KEY` | Chave da OpenAI (obrigatória se `MODEL_PROVIDER=openai`) | — |
| `OPENAI_EMBEDDING_MODEL` | Modelo de embeddings da OpenAI | `text-embedding-3-small` |
| `OPENAI_LLM_MODEL` | Modelo de chat da OpenAI | `gpt-4o-mini` |
| `GOOGLE_API_KEY` | Chave do Google Gemini (obrigatória se `MODEL_PROVIDER=google`) | — |
| `GOOGLE_EMBEDDING_MODEL` | Modelo de embeddings do Gemini | `gemini-embedding-001` |
| `GOOGLE_LLM_MODEL` | Modelo de chat do Gemini | `gemini-3.5-flash-lite` |
| `DATABASE_URL` | Conexão do PostgreSQL (driver `psycopg`) | `postgresql+psycopg://postgres:postgres@localhost:5432/rag` |
| `PG_VECTOR_COLLECTION_NAME` | Nome da collection no pgVector | `document_chunks` |
| `PDF_PATH` | Caminho do PDF a ingerir | `document.pdf` |

Chaves de API:

- OpenAI: <https://platform.openai.com/api-keys>
- Google Gemini: <https://aistudio.google.com/apikey>

## Ordem de execução

### 1. Subir o banco de dados

```bash
docker compose up -d
```

O Compose sobe o PostgreSQL com a imagem `pgvector/pgvector:pg17` e executa um
serviço único que cria a extensão `vector`. Confira com:

```bash
docker compose ps
```

### 2. Executar a ingestão do PDF

```bash
python src/ingest.py
```

Saída esperada (valores aproximados):

```
Carregando PDF: document.pdf
Páginas carregadas: 34
Chunks gerados: 125 (chunk_size=1000, overlap=150)
Gerando embeddings e gravando no PostgreSQL/pgVector...
Gravados 90/125 chunks.
Gravados 125/125 chunks.
Ingestão concluída: 125 chunks na collection 'document_chunks'.
```

> O free tier do Gemini conta cada texto enviado como uma requisição por minuto.
> Quando o limite é atingido, a ingestão espera o tempo indicado pela API e
> tenta novamente automaticamente (os lotes são de 90 chunks).

> A ingestão usa `pre_delete_collection=True`: a cada execução a collection é
> apagada e recriada. Isso evita documentos duplicados e permite trocar o modelo
> de embeddings sem precisar apagar o volume do banco.

### 3. Rodar o chat

```bash
python src/chat.py
```

Exemplo:

```
Chat de busca semântica sobre o PDF.
Digite sua pergunta ou 'sair' para encerrar.

PERGUNTA: Qual o faturamento da Empresa SuperTechIABrazil?
RESPOSTA: O faturamento da Empresa SuperTechIABrazil é de R$ 10.000.000,00.

---
```

Pergunta fora do contexto:

```
PERGUNTA: Quantos clientes temos em 2024?
RESPOSTA: Não tenho informações necessárias para responder sua pergunta.

---
```

Comandos `sair`, `exit`, `quit`, `q`, `Ctrl+D` e `Ctrl+C` encerram o chat.

Também é possível fazer uma pergunta única:

```bash
python src/search.py "Qual o faturamento da Empresa SuperTechIABrazil?"
```

## Como funciona

### Ingestão (`src/ingest.py`)

1. `PyPDFLoader` lê o PDF com `extraction_mode="layout"`. O modo layout preserva
   as linhas da tabela; o modo padrão separa nome, faturamento e ano em linhas
   diferentes, prejudicando a busca.
2. `RecursiveCharacterTextSplitter` divide o texto em chunks de 1000 caracteres
   com overlap de 150.
3. Os chunks são vetorizados pelo modelo de embeddings do provedor configurado,
   em lotes, com nova tentativa automática quando a API retorna limite de uso.
4. `PGVector` grava texto, metadados (`source` e `page`) e vetores no PostgreSQL.

### Busca (`src/search.py`)

1. A pergunta é vetorizada pelo mesmo modelo de embeddings.
2. `similarity_search_with_score(pergunta, k=10)` retorna os 10 trechos mais
   próximos.
3. O contexto é montado com o cabeçalho da tabela do PDF
   (`Nome da empresa | Faturamento | Ano de fundação`) seguido dos 10 trechos.
   Como os chunks contêm apenas linhas da tabela, o cabeçalho permite à LLM
   interpretar corretamente o significado de cada coluna.
4. A LLM responde com `temperature=0`, restrita ao contexto. Se a resposta não
   estiver no material, o modelo devolve a frase padrão:

   > Não tenho informações necessárias para responder sua pergunta.

O prompt utilizado é exatamente o do enunciado do desafio, definido em
`src/search.py` (`PROMPT_TEMPLATE`).

## Troca de provedor e dimensão dos vetores

| Provedor | Embedding padrão | Dimensões |
| --- | --- | --- |
| OpenAI | `text-embedding-3-small` | 1536 |
| Google Gemini | `gemini-embedding-001` | 3072 |

A collection é criada com a dimensão do modelo usado na primeira ingestão. Para
trocar de provedor:

1. Altere `MODEL_PROVIDER` e preencha a chave correspondente no `.env`.
2. Rode novamente `python src/ingest.py`.

A collection antiga é recriada automaticamente com a nova dimensão. Se você
tentar consultar antes de reingerir, o chat exibe uma mensagem pedindo para rodar
a ingestão novamente.

Também é possível trocar o modelo dentro do mesmo provedor pelas variáveis
`OPENAI_EMBEDDING_MODEL` / `GOOGLE_EMBEDDING_MODEL`, desde que a ingestão seja
refeita.

## Perguntas de teste

Dentro do contexto:

- `Qual o faturamento da Empresa SuperTechIABrazil?`
- `Qual o ano de fundação da SuperTechIABrazil?`
- `Qual o faturamento da Alfa Energia S.A.?`

Fora do contexto (devem retornar a frase padrão):

- `Quantos clientes temos em 2024?`
- `Qual é a capital da França?`
- `Você acha isso bom ou ruim?`

## Solução de problemas

**`connection refused` ao rodar a ingestão**

O banco ainda não subiu. Verifique `docker compose ps` e aguarde o healthcheck.

**Porta 5432 já está em uso**

Altere o mapeamento no `docker-compose.yml` para `"5433:5432"` e ajuste
`DATABASE_URL` para `...@localhost:5433/rag`.

**`extension "vector" is not available`**

O serviço `bootstrap_vector_ext` não executou. Rode:

```bash
docker compose up -d
docker compose logs bootstrap_vector_ext
```

**`Variável OPENAI_API_KEY não configurada` (ou `GOOGLE_API_KEY`)**

Confira se o `.env` existe na raiz do projeto e se a chave do provedor escolhido
em `MODEL_PROVIDER` está preenchida.

**A resposta é sempre “Não tenho informações...”**

A collection pode estar vazia. Execute `python src/ingest.py` antes de abrir o
chat. Confira também a variável `PG_VECTOR_COLLECTION_NAME`.

**Erro 429 / `quota exceeded` no Gemini**

O free tier limita requisições por minuto e por dia. A ingestão espera e tenta
novamente quando o limite por minuto é atingido. Se a cota diária acabar, use
outro modelo em `GOOGLE_LLM_MODEL`/`GOOGLE_EMBEDDING_MODEL` (refazendo a
ingestão) ou aguarde a renovação da cota.

**Erro de dimensão ao consultar**

O modelo de embeddings mudou depois da ingestão. Rode `python src/ingest.py`
para recriar a collection com a nova dimensão.

**Recomeçar do zero**

```bash
docker compose down -v
docker compose up -d
python src/ingest.py
```

O `-v` remove o volume do PostgreSQL, apagando todos os dados do banco.

## Requisitos do desafio atendidos

- [x] PDF dividido em chunks de 1000 caracteres com overlap de 150.
- [x] Cada chunk convertido em embedding e salvo no PostgreSQL com pgVector.
- [x] Busca dos 10 resultados mais relevantes (`k=10`).
- [x] Prompt do enunciado utilizado integralmente.
- [x] CLI com `PERGUNTA:` / `RESPOSTA:` e resposta padrão para perguntas fora do contexto.
- [x] Banco executado com Docker Compose.
- [x] Suporte a OpenAI e Google Gemini.