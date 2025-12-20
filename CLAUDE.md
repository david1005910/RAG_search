# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a **Medical/Scientific Paper RAG System** with multilingual support. It searches papers from arXiv and PubMed, downloads PDFs, extracts text, summarizes papers using OpenAI, builds vector embeddings with medical-specialized models (BioBERT/PubMedBERT), and enables semantic search queries.

## Running the System

### Quick Start (Python 3.11 with Conda)
```bash
# Activate Python 3.11 conda environment
source /usr/local/Caskroom/miniconda/base/bin/activate ./venv311

# Run interactive RAG system
python run_rag.py
```

### Fresh Installation
```bash
# Create conda environment with Python 3.11
conda create -y -p ./venv311 python=3.11

# Activate environment
source /usr/local/Caskroom/miniconda/base/bin/activate ./venv311

# Install dependencies
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
pip install sentence-transformers "transformers<4.50" "numpy<2"
pip install arxiv biopython pypdf2 pdfplumber langchain langchain-community langchain-core faiss-cpu requests xmltodict openai python-dotenv

# Run
python run_rag.py
```

## API Key Management (.env file)

API keys are managed via `.env` file in the project root:

```env
# PubMed API (optional - faster requests)
PUBMED_API_KEY=your_key_here
PUBMED_EMAIL=your_email@example.com

# OpenAI API (paper summarization & Korean→English translation)
OPENAI_API_KEY=sk-...

# Anthropic API (future Claude integration)
ANTHROPIC_API_KEY=sk-ant-...
```

Get API keys:
- PubMed: https://www.ncbi.nlm.nih.gov/account/settings/
- OpenAI: https://platform.openai.com/api-keys

## Key Features

### Multilingual Support
- **Korean search**: Korean queries are automatically translated to English for search
- **Korean responses**: When query is Korean, all responses are in Korean
- Example: "당뇨병 치료" → searches "diabetes treatment" → responds in Korean

### Paper Summarization
- Uses OpenAI GPT-3.5 to summarize papers after download
- Summarizes: title, authors, abstract, and full content
- Output format: Research Objective, Key Methods, Main Results, Conclusion

### Medical Embedding Models
| Model | Description |
|-------|-------------|
| `biobert` | BioBERT (default) - medical/biology specialized |
| `pubmedbert` | PubMedBERT - optimized for PubMed papers |
| `scibert` | SciBERT - general science |
| `minilm` | MiniLM - fast, general purpose |

## Architecture

```
User Query (Korean/English)
    ↓
Language Detection & Translation (if Korean)
    ↓
Paper Search (arXiv/PubMed)
    ↓
PDF Download
    ↓
Text Extraction
    ↓
Paper Summarization (OpenAI)
    ↓
Chunking → Embedding (BioBERT/PubMedBERT)
    ↓
FAISS Vector Store
    ↓
Interactive Q&A (with Korean support)
```

### Core Classes

| Class | Purpose |
|-------|---------|
| `Config` | Interactive setup, .env API key loading |
| `PaperSearcher` | arXiv and PubMed API search |
| `PDFDownloader` | PDF download with abstract fallback |
| `TextExtractor` | PDF/TXT text extraction |
| `PaperSummarizer` | OpenAI-based paper summarization |
| `EmbeddingModelFactory` | Medical embedding model factory |
| `RAGSystem` | FAISS vector store and similarity search |

### Storage Directories

- `./papers/` - Downloaded PDFs and text files
- `./vectorstore/` - Persisted FAISS index
- `./.env` - API keys (not committed to git)

## Interactive Q&A Commands

- Type question and press Enter
- Add `k=5` to change number of results
- `q`, `quit`, `exit`, `종료`, `끝` to exit

## Dependency Notes

- **Python 3.11**: Required for BioBERT/PubMedBERT support
- **transformers<4.50**: Avoids PyTorch 2.6+ security check requirement
- **numpy<2**: Required for PyTorch 2.2.x compatibility
- **python-dotenv**: For .env file loading
