# Papers QA Agent

A LangChain-powered question-answering system for scientific papers using Ollama LLM and ChromaDB for vector storage.

## Features

* Load and process PDF papers automatically
* Smart document chunking and embedding generation
* Interactive Q&A interface for paper content
* Vector similarity search using ChromaDB
* Progress tracking for document processing

## Prerequisites

* Python 3.8+
* Ollama running locally
* PDF papers to analyze

## Installation

1. Clone the repository:

```bash
git clone <repository-url>
cd papers-qa-agent
```

2. Install dependencies :

```bash
pip install -r requirements.txt
```

4. Configure environment variables in .env:

```plaintext
PROJECT_NAME=scientific_papers_qa
PAPERS_DIR=./papers
OLLAMA_MODEL=llama3.2
CHROMA_DB_DIR=./chroma_db
OLLAMA_BASE_URL=http://localhost:11434
```


## Usage

1. Add PDF papers to the papers directory
2. Run the application:

```bash
python __main__.py
```

### Available Commands

* `help`: Show available commands
* `metadata`: Display information about loaded papers
* `exit`: Quit the application
