# RAG Playground Backend

Local-first RAG experimentation platform built with LangChain v1.0+, FastAPI, and Ollama.

## Implementation Status

### Phase 0: Foundation - COMPLETED ✅

Foundational infrastructure:
- ✅ Project setup with `uv` package manager
- ✅ LangChain v1.0+ integration (latest versions)
- ✅ FastAPI application structure
- ✅ Configuration management with pydantic-settings
- ✅ Ollama health checks with fail-fast startup
- ✅ LlmLockManager for resource safety
- ✅ SQLite database with SQLModel ORM
- ✅ ChromaDB PersistentClient integration
- ✅ InMemoryStore for parent document storage
- ✅ BM25 manager with on-demand rebuild
- ✅ PDF processing with PyMuPDFLoader

### Phase 1: Foundational Backend - COMPLETED ✅

Core RAG functionality:
- ✅ Configuration management enhancement (RAG parameters)
- ✅ Three-layer validation system (Indexing, Pipeline, Advanced)
- ✅ Atomic context limiter (drops full chunks)
- ✅ Document ingestion service (standard + parent document)
- ✅ Documents API (upload, list, get, delete)
- ✅ Core RAG techniques (Basic, Fusion, Reranking)
- ✅ LCEL-based pipeline (composable chains)
- ✅ Query endpoint with technique selection
- ✅ Result management & scoring (semantic variance)
- ✅ Error handling & API route wiring

**Available Endpoints**: 15 routes at `/api/v1/*` (see `/docs` for documentation)

## Prerequisites

1. **Python 3.11+** (currently using Python 3.13.3)
2. **Ollama** - Local LLM service
3. **uv** - Python package manager

## Setup Instructions

### 1. Install Ollama

```bash
# Install Ollama from https://ollama.ai
# Or use your package manager

# Pull required models
ollama pull nomic-embed-text
ollama pull llama3.2
```

### 2. Start Ollama Service

#### ⚠️ CRITICAL: Configure Ollama for Parallel Execution

**These settings are REQUIRED for optimal performance.** Without them, parallel operations (HyDE + Retrieval) will run sequentially, causing 2-3x slower queries.

```bash
# Set these BEFORE starting Ollama (REQUIRED for best performance)
export OLLAMA_NUM_PARALLEL=4          # Allow 4 parallel requests (HyDE + Retrieval + multiple users)
export OLLAMA_MAX_LOADED_MODELS=2     # Keep both 3B and 20B models in memory

# Then start Ollama
ollama serve
```

**Why these settings matter:**

| Setting | Purpose | Impact if not set |
|---------|---------|-------------------|
| `OLLAMA_NUM_PARALLEL=4` | Allows HyDE (3B model) and Retrieval to run in parallel | HyDE + Retrieval runs sequentially: **23s instead of 11s** |
| `OLLAMA_MAX_LOADED_MODELS=2` | Keeps both models in memory | Model reload delays: **+2-5s per query** |

**Performance Impact:**
- ✅ **With settings**: `basic_rag + hyde + reranking` = ~8-10s
- ❌ **Without settings**: `basic_rag + hyde + reranking` = ~20-23s

**Requirements:**
- Pull both models: `ollama pull llama3.2:3b` and `ollama pull gpt-oss:20b-cloud`
- Sufficient RAM/VRAM: 16GB+ RAM or 8GB+ VRAM recommended
- Mac users: Apple Silicon (M1/M2/M3) handles this well with unified memory

### 3. Install Dependencies

```bash
# Clone the repository (if not already)
cd rag-lab

# Install all dependencies with uv
uv sync
```

### 4. Configure Environment

```bash
# Copy example environment file
cp .env.example .env

# Edit .env if needed (defaults should work for local development)
```

### 5. Run the Application

```bash
# Start the FastAPI server
uv run uvicorn app.main:app --reload

# The application will:
# 1. Create database tables (playground.db)
# 2. Verify Ollama is running and models are available
# 3. Start the API server on http://localhost:8000
```

## Quick Start

```bash
# 1. Ensure Ollama is running
ollama serve

# 2. Start the server (in another terminal)
uv run uvicorn app.main:app --reload

# 3. View API documentation
open http://localhost:8000/docs

# 4. Upload a document
curl -X POST "http://localhost:8000/api/v1/documents/upload" \
  -F "file=@your_document.pdf" \
  -F "chunk_size=1024" \
  -F "chunk_overlap=200" \
  -F "chunking_strategy=standard"

# 5. Query the document (use document_id from upload response)
curl -X POST "http://localhost:8000/api/v1/rag/query" \
  -H "Content-Type: application/json" \
  -d '{
    "document_id": "your-document-id-here",
    "query": "What is this document about?",
    "techniques": ["basic_rag"],
    "query_params": {"top_k": 5, "temperature": 0.7}
  }'
```

## API Endpoints

### System
- `GET /` - Root endpoint
- `GET /health` - Health check (verifies Ollama connectivity)

### Documents (Phase 1)
- `POST /api/v1/documents/upload` - Upload PDF with indexing configuration
- `GET /api/v1/documents` - List documents (optionally filtered by session)
- `GET /api/v1/documents/{doc_id}` - Get document details
- `DELETE /api/v1/documents/{doc_id}` - Delete document

### RAG Queries (Phase 1)
- `POST /api/v1/rag/query` - Execute RAG query with technique selection
- `POST /api/v1/rag/validate` - Validate technique combination

### Results (Phase 1)
- `GET /api/v1/results/{session_id}` - Get all results for session
- `GET /api/v1/results/compare?result_ids=id1,id2,...` - Compare results
- `GET /api/v1/results/detail/{result_id}` - Get result details

**Full API Documentation**: Visit `http://localhost:8000/docs` when server is running

## Project Structure

```
rag-lab/
├── app/
│   ├── main.py              # FastAPI application entry point
│   ├── core/                # Core functionality
│   │   ├── config.py        # Configuration management
│   │   ├── health.py        # Ollama health checks
│   │   ├── concurrency.py   # LlmLockManager
│   │   └── dependencies.py  # Dependency injection
│   ├── db/                  # Database layer
│   │   ├── database.py      # SQLModel engine
│   │   ├── models.py        # Database models
│   │   └── repositories.py  # Data access layer
│   ├── services/            # Business logic services
│   │   ├── vectorstore.py   # ChromaDB & LocalFileStore
│   │   ├── bm25_manager.py  # BM25 index management
│   │   └── ingestion.py     # PDF processing
│   ├── models/              # Pydantic schemas (Phase 1)
│   ├── api/                 # API routes (Phase 1)
│   ├── rag/                 # RAG pipeline (Phase 1)
│   └── utils/               # Utilities
├── chromadb/                # Vector store data (gitignored)
├── storage/parents/         # Parent document store (gitignored)
├── uploads/                 # Uploaded PDFs (gitignored)
├── playground.db            # SQLite database (gitignored)
├── pyproject.toml           # Project dependencies
└── .env                     # Environment configuration

```

## Technology Stack

### Core Framework
- **FastAPI** - Modern async web framework
- **SQLModel** - SQL database ORM
- **Pydantic Settings** - Configuration management

### LangChain Ecosystem (v1.0+)
- **langchain** - Core orchestration framework
- **langchain-ollama** (v1.0.1) - Ollama integration
- **langchain-chroma** - ChromaDB integration
- **langchain-community** - Document loaders and utilities
- **langchain-experimental** - Advanced features
- **langchain-text-splitters** - Text chunking strategies

### Vector & Search
- **ChromaDB** - Embedded vector database
- **rank-bm25** - BM25 keyword search
- **sentence-transformers** - Embedding models

### Additional Libraries
- **httpx** - Async HTTP client
- **pymupdf** - PDF processing
- **scikit-learn** - ML utilities
- **tavily-python** - Web search (Phase 3)

## Key Features

### Resource Safety
- **LlmLockManager**: Prevents concurrent Ollama calls that could crash the system
- Querying is rejected with 503 when indexing is active

### Health Checks
- Application verifies Ollama connectivity at startup
- Fails fast with clear error messages if dependencies unavailable
- Runtime `/health` endpoint for monitoring

### Database
- SQLite for zero-setup persistence
- Automatic table creation on startup
- Document, QueryResult, and Session models

### Vector Storage
- ChromaDB in embedded mode (no server required)
- One collection per document: `doc_{document_id}`
- LocalFileStore for parent document retrieval

### BM25 Index
- On-demand rebuild from ChromaDB (no persistence)
- In-memory LRU cache for performance
- Fast: <1s for 1000 chunks

## Development

### Running Tests

```bash
# Run with uv
uv run pytest

# With coverage
uv run pytest --cov=app
```

### Checking Types

```bash
# Install mypy
uv add --dev mypy

# Run type checking
uv run mypy app/
```

### Code Formatting

```bash
# Install black and ruff
uv add --dev black ruff

# Format code
uv run black app/
uv run ruff check app/ --fix
```

## Next Steps

See the phase documents for implementation roadmap:

- **Phase 1**: Core RAG pipeline, document ingestion, query endpoints
- **Phase 2**: HyDE, Semantic Chunking, Contextual Compression
- **Phase 3**: Self-RAG, CRAG, Adaptive Retrieval

## Troubleshooting

### Application won't start

**Error**: "Cannot connect to Ollama"

**Solution**: Ensure Ollama is running (`ollama serve`) and models are pulled

### Database errors

**Error**: "Unable to open database file"

**Solution**: Ensure the application has write permissions in the project directory

### Import errors

**Error**: "No module named 'langchain_ollama'"

**Solution**: Run `uv sync` to install all dependencies

### Slow HyDE Performance

**Symptom**: `basic_rag + hyde` takes more than 10 seconds (should be 3-4s)

**Diagnosis Steps**:

1. **Check Ollama Environment Variables** (MOST COMMON ISSUE):
   ```bash
   # Check if variables are set in current shell
   echo $OLLAMA_NUM_PARALLEL
   echo $OLLAMA_MAX_LOADED_MODELS
   
   # If not set or empty, export them BEFORE starting Ollama
   export OLLAMA_NUM_PARALLEL=4
   export OLLAMA_MAX_LOADED_MODELS=2
   
   # CRITICAL: Restart Ollama after setting variables
   # Stop Ollama (Ctrl+C), then:
   ollama serve
   ```

2. **Verify Models are Loaded**:
   ```bash
   # Check if models are in memory
   ollama ps
   # Should show: llama3.2:3b and nomic-embed-text
   ```

3. **Check Server Logs**:
   - Look for "HyDE expansion completed in Xms" in server logs
   - If >10s, you'll see a warning message
   - Warning indicates Ollama is processing requests sequentially

4. **Verify Model Configuration**:
   - Check logs for "HyDE LLM configured: model=llama3.2:3b"
   - Should use 3B model, not 20B model

**Expected Performance**:
- ✅ **With proper config**: `basic_rag + hyde` = 3-4s
- ❌ **Without config**: `basic_rag + hyde` = 20-25s

**Note**: Even if models are loaded (`ollama ps` shows them), you MUST set environment variables BEFORE starting Ollama for parallel processing to work.

## License

See LICENSE file for details.

## Documentation

- [IMPLEMENTATION_PLAN.md](IMPLEMENTATION_PLAN.md) - Overall architecture and design
- [PHASE_0_FOUNDATION.md](PHASE_0_FOUNDATION.md) - This phase's detailed spec
- [PHASE_1_BACKEND.md](PHASE_1_BACKEND.md) - Next phase implementation
