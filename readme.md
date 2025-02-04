# Papers QA Agent

A LangChain-powered question-answering system for scientific papers using Ollama LLM and ChromaDB for vector storage.

## Features

* Load and process PDF papers automatically
* Smart document chunking and embedding generation
* Interactive Q&A command line interface 
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

2. Setup and dependencies :

```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install requirements
pip install -r requirements.txt
```

4. Configure environment variables in .env:

```plaintext
PROJECT_NAME=scientific_papers_qa
PAPERS_DIR=./papers
OLLAMA_MODEL=deepseek-r1:14b 
OLLAMA_EMBEDDING_MODEL=snowflake-arctic-embed:335m-l-fp16
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

* `audio`: Use the audio to text STT feature
* `help`: Show available commands
* `metadata`: Display information about loaded papers
* `exit`: Quit the application

### Additional notes:

pyaudio might require additional system-level dependencies:

* On Windows: No additional requirements
* On Linux: `sudo apt-get install python3-pyaudio`
* On macOS: `brew install portaudio`
