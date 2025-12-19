# RAG Playground Backend - Final Implementation Plan

**CRITICAL**: This plan uses **LangChain v1.0.0+**. Before implementation, read:
- [LangChain Documentation](https://python.langchain.com/docs/get_started/introduction)
- [LangChain RAG Techniques](https://python.langchain.com/docs/use_cases/question_answering/)
- [LangChain Experimental Features](https://python.langchain.com/docs/experimental/)
- [LCEL (LangChain Expression Language) Guide](https://python.langchain.com/docs/expression_language/)

This ensures proper understanding of modern LangChain patterns, LCEL composition, and built-in RAG techniques before coding.

---

## Implementation Phases

This implementation plan is split into focused phase documents for better organization and step-by-step execution:

- **[Phase 0: Local Architecture Foundation](PHASE_0_FOUNDATION.md)** - Project setup, dependencies, Ollama integration, database, ChromaDB, and core infrastructure
- **[Phase 1: Foundational Backend](PHASE_1_BACKEND.md)** - Core RAG pipeline, document ingestion, validation system, query endpoints, and result management
- **[Phase 2: Query & Context Enhancements](PHASE_2_ENHANCEMENTS.md)** - HyDE, Semantic Chunking, Contextual Compression, Contextual Headers, and enhanced scoring
- **[Phase 3: Advanced Architectures](PHASE_3_ADVANCED.md)** - Self-RAG, Corrective RAG (CRAG), Adaptive Retrieval, and advanced flow controllers

**Start with Phase 0** and proceed sequentially. Each phase document contains detailed to-do lists, implementation steps, and success criteria.

---

## 1. Project Purpose & Philosophy

This project builds an interactive, **local-first** RAG experimentation platform. It allows developers to **upload documents once** and **A/B test different retrieval strategies** without writing code.

**Core Design Principles:**
1. **Democratization**: Accessible UI for complex RAG pipelines
2. **Local & Lightweight**: Runs on a single machine (CPU/Consumer GPU) using Ollama, SQLite, and Chroma (Embedded)
3. **Destructive vs. Non-Destructive**: Clear separation between "Indexing" (requires rebuild) and "Querying" (instant change) parameters
4. **Resource Safety**: Includes concurrency locks and RAM guards to prevent local machine crashes
5. **Traceability**: Every answer includes exact Page/Line numbers from the PDF
6. **Comparison**: Users can prove that "Strategy B" is better than "Strategy A" using Semantic Variance scoring

## Key Design Decisions

- **LLM Provider**: LangChain Ollama (local execution via Ollama API)
- **Database**: SQLite with SQLModel (zero-setup, file-based persistence)
- **Embeddings**: Ollama Nomic Embeddings (local, no API costs)
- **Vector Store**: ChromaDB PersistentClient (embedded mode, no server)
- **Chunk Size**: Configurable per document upload (not a technique - parameter)
- **Chunking Strategy**: Index-time parameter (requires re-upload to change)
- **Web Search**: Tavily (recommended) or DuckDuckGo (fallback) for CRAG Phase 3
- **Result Management**: Session-based storage with scoring and comparison
- **Streaming**: Not required - process and return complete response
- **Authentication**: No auth for MVP - anonymous sessions
- **Architecture**: Local/lightweight playground - no external services required (except Ollama)

## 2. Tech Stack (Modern Ecosystem)

**CRITICAL**: Use **LangChain v1.0.0+** (not v0.x). This ensures compatibility with modern LCEL patterns and experimental features.

- **Runtime**: Python 3.11+
- **API**: FastAPI (Async)
- **Database**: SQLite + SQLModel
- **Orchestration**: `langchain>=1.0.0`, `langchain-core`
- **Components**: 
  - `langchain-community` (document loaders, integrations)
  - `langchain-ollama` (Ollama LLM and embeddings)
  - `langchain-chroma` (ChromaDB integration)
  - `langchain-text-splitters` (chunking strategies)
  - `langchain-experimental` (SemanticChunker, advanced techniques)
- **Vector Store**: ChromaDB (PersistentClient - Local Mode)
- **Key/Value Store**: LangChain `LocalFileStore` (for Parent Document Retriever)
- **Search**: `tavily-python` (Primary for CRAG), `duckduckgo-search` (Fallback)
- **Math/Ranking**: `rank-bm25`, `scikit-learn` (for Semantic Variance), `sentence-transformers`
- **Process Management**: `uv` (Package Manager)
- **File Uploads**: `python-multipart`

## Architecture Overview

### Local Execution Model

This playground runs entirely locally with minimal setup:

1. **SQLite Database**: Single file (`playground.db`) - no server required
2. **ChromaDB PersistentClient**: Embedded mode, stores vectors in `./chromadb/` folder
3. **Ollama Service**: Must be running locally (`ollama serve`)
4. **BM25 Indices**: Rebuilt on-demand from ChromaDB chunks (with in-memory LRU cache)
5. **File Storage**: PDFs stored in `./uploads/` directory
6. **Parent Document Store**: LocalFileStore at `./storage/parents/` for full context windows

### Critical Architecture Components

#### A. Global Resource Lock (The "Ollama Guard")

**Component**: `app.core.concurrency.LlmLockManager`

**Problem**: Concurrent Ollama calls (indexing + querying) can crash the local machine or cause resource exhaustion.

**Solution**: Treat LLM as a singleton resource with `asyncio.Lock()`:
- **Indexing** (background) acquires the lock
- **Querying** (foreground) attempts to acquire
- **Collision**: If Indexing is active, Query returns `503 Service Busy (Indexing in Progress)`

**Implementation**:
```python
class LlmLockManager:
    def __init__(self):
        self._lock = asyncio.Lock()
        self._indexing_active = False
    
    async def acquire_for_indexing(self):
        await self._lock.acquire()
        self._indexing_active = True
    
    async def acquire_for_query(self) -> bool:
        if self._indexing_active:
            return False  # Reject query
        await self._lock.acquire()
        return True
```

#### B. Parent Document Storage

**Strategy**: Store small chunks in ChromaDB for vector similarity, but keep full parent documents in LocalFileStore for context retrieval.

- **Small Chunks**: Stored in ChromaDB (`./chromadb/`) for vector similarity
- **Parent Docs**: Stored in LocalFileStore (`./storage/parents/`) using LangChain's native implementation
- **Benefit**: Retrieves full context windows without bloating the Vector DB

#### C. Atomic Context Limiter

**Component**: `app.rag.limiter.AtomicContextLimiter`

**Problem**: Hard-cutting text at character limits risks broken sentences and context loss.

**Solution**: Drop full chunks instead of slicing:
1. Determines available budget: `Total - (SystemPrompt + Query)`
2. Adds full chunks until budget is near full
3. **Drops** the lowest-ranked chunk entirely rather than slicing it
4. Preserves sentence integrity and context coherence

### Health Checks

**Critical**: Application must validate dependencies at startup:

- **Ollama Health Check**:
  - Verify Ollama API accessible (`http://localhost:11434`)
  - Verify embedding model available (`nomic-embed-text`)
  - Fail fast with clear error if unavailable
  - Provide setup instructions in error message

- **ChromaDB**: Auto-initializes on first use (no check needed)

- **SQLite**: Auto-creates database file if missing (no check needed)

## Curated RAG Techniques

### Three-Layer Validation System

Techniques are organized into three layers with specific combination rules:

**The Golden Rule**: Pick **ONE** from Layer 1, **ANY** from Layer 2, and **MAX ONE** from Layer 3.

---

### Layer 1: Indexing Strategy (Mutually Exclusive - Upload Time)

*These happen at **Upload Time**. They define how text is physically stored. Select exactly ONE:*

| # | Technique | Impact | Time | Notes |

|---|-----------|--------|------|-------|

| Default | **Standard Chunking** | ⭐⭐⭐ | - | Baseline recursive character split. Fast, robust. |
| **Parent Document** | *LangChain Native* | ⭐⭐⭐⭐⭐ | 2d | High retrieval accuracy; preserves full context. Requires dual storage (Vector + Key/Value). |
| 5 | [Semantic Chunking](https://github.com/NirDiamant/RAG_TECHNIQUES/blob/main/all_rag_techniques/semantic_chunking.ipynb) | ⭐⭐⭐⭐ | 2-3d | Splits by meaning using embeddings. **Slower**: Requires embedding during ingestion. |
| 7 | [Contextual Headers](https://github.com/NirDiamant/RAG_TECHNIQUES/blob/main/all_rag_techniques/contextual_chunk_headers.ipynb) | ⭐⭐⭐⭐ | 2-3d | Preserves document hierarchy. **Very Slow**: LLM call for every chunk. |
| 8 | [Proposition Chunking](https://github.com/NirDiamant/RAG_TECHNIQUES/blob/main/all_rag_techniques/proposition_chunking.ipynb) | ⭐⭐⭐⭐⭐ | 3-4d | Atomic fact precision. **Critical Speed Issue**: Phase 3 only. |

**Constraints**:

- Cannot combine Semantic + Proposition (different storage structures)
- Parent Document is mutually exclusive with other strategies (uses different storage pattern)
- Contextual Headers can be combined with Standard or Semantic
- Proposition Chunking is mutually exclusive with Headers

**Performance Warnings**:

- **Contextual Headers**: LLM generates title for every chunk. 20-page PDF ≈ 30 minutes on local LLM
- **Proposition Chunking**: LLM rewrites every sentence. Extremely slow on local LLM
- **Recommendation**: Mark these as "Experimental/Slow" in UI with progress indicators

---

### Layer 2: Pipeline Components (Mix & Match - Query Time)

*These happen at **Query Time**. These are modular blocks that can be chained. Select ANY combination:*

| # | Technique | Logic Position | Impact | Time | Notes |

|---|-----------|----------------|--------|------|-------|

| 4 | [HyDE](https://github.com/NirDiamant/RAG_TECHNIQUES/blob/main/all_rag_techniques/HyDe_Hypothetical_Document_Embedding.ipynb) | Step 1 (Pre-Retrieval) | ⭐⭐⭐⭐ | 2d | Query expansion - universally compatible |

| 1 | [Basic RAG](https://github.com/NirDiamant/RAG_TECHNIQUES/blob/main/all_rag_techniques/simple_rag.ipynb) | Step 2 (Retrieval) | ⭐⭐⭐⭐ | 1-2d | Vector search only (base choice) |

| 2 | [Fusion Retrieval](https://github.com/NirDiamant/RAG_TECHNIQUES/blob/main/all_rag_techniques/fusion_retrieval.ipynb) | Step 2 (Retrieval) | ⭐⭐⭐⭐⭐ | 2-3d | BM25 + Vector (replaces Basic RAG) |

| 3 | [Reranking](https://github.com/NirDiamant/RAG_TECHNIQUES/blob/main/all_rag_techniques/reranking.ipynb) | Step 3 (Post-Retrieval) | ⭐⭐⭐⭐⭐ | 2d | Re-orders top results (always recommended) |

| 6 | [Contextual Compression](https://github.com/NirDiamant/RAG_TECHNIQUES/blob/main/all_rag_techniques/contextual_compression.ipynb) | Step 4 (Post-Rerank) | ⭐⭐⭐ | 2d | Shortens text passed to LLM |

**Constraints**:

- Must choose **EITHER** `#1 Basic RAG` **OR** `#2 Fusion Retrieval` (not both)
- Fusion Retrieval *contains* Basic RAG (BM25 + Vector)
- Reranking requires retrieval technique (Basic or Fusion)
- Compression requires retrieval technique
- HyDE works with everything (universally compatible)

**Implementation Notes**:

- **Fusion Retrieval**: Must use LangChain's `EnsembleRetriever`
- **BM25 Rebuild**: Rebuild BM25 index from ChromaDB chunks on-demand (fast: <1s for 1000 chunks). Use in-memory LRU cache to avoid repeated rebuilds.
- **Reranking**: Uses sentence-transformers CrossEncoder (⚠️ Requires 1GB+ RAM in addition to Ollama)

---

### Layer 3: Advanced Flow Controllers (Mutually Exclusive - Query Time)

*These are complex architectures that take over the entire control loop. Select MAX ONE (or None):*

| # | Technique | Impact | Time | Notes |

|---|-----------|--------|------|-------|

| 9 | [Self-RAG](https://github.com/NirDiamant/RAG_TECHNIQUES/blob/main/all_rag_techniques/self_rag.ipynb) | ⭐⭐⭐⭐⭐ | 3-4d | LLM decides IF retrieval needed + relevance evaluation |

| 10 | [Corrective RAG (CRAG)](https://github.com/NirDiamant/RAG_TECHNIQUES/blob/main/all_rag_techniques/crag.ipynb) | ⭐⭐⭐⭐⭐ | 3-5d | Evaluates retrieval + web search fallback |

| 11 | [Adaptive Retrieval](https://github.com/NirDiamant/RAG_TECHNIQUES/blob/main/all_rag_techniques/adaptive_retrieval.ipynb) | ⭐⭐⭐⭐ | 2-3d | Routes queries to different strategies |

**Constraints**:

- Cannot combine Self-RAG + CRAG (logic collision - both have critique loops)
- Adaptive Retrieval is best treated as standalone "Auto" mode
- All Layer 3 techniques are compatible with Layer 1 & 2

**Performance Notes**:

- **Self-RAG**: 3x latency (3 LLM calls per query: retrieve decision, generation, critique)
- **CRAG**: Requires web search tool (Tavily recommended over DuckDuckGo for stability)

---

### Recommended "Recipes" (Presets for Users)

1. **"Quality King"** (Slow but Smart):

   - Index: `#5 Semantic Chunking`
   - Pipeline: `#4 HyDE` + `#2 Fusion` + `#3 Reranking`
   - *Why*: Best possible context understanding

2. **"Fact Checker"** (For messy docs):

   - Index: `#7 Contextual Headers`
   - Pipeline: `#10 CRAG`
   - *Why*: Ensures data correctness via web search fallback

3. **"Summary"** (Fast):

   - Index: `Standard`
   - Pipeline: `#2 Fusion` + `#6 Contextual Compression`
   - *Why*: Fits lots of context into small prompts

4. **"Balanced"** (Recommended starting point):

   - Index: `Standard`
   - Pipeline: `#2 Fusion` + `#3 Reranking`
   - *Why*: Good quality-to-speed ratio

## Project Structure

```
rag-playground-backend/
├── app/
│   ├── __init__.py
│   ├── main.py                      # FastAPI app + startup health checks
│   ├── api/
│   │   ├── __init__.py
│   │   ├── v1/
│   │   │   ├── __init__.py
│   │   │   ├── documents.py         # POST /upload, GET /list, DELETE /{doc_id}
│   │   │   ├── rag.py               # POST /query
│   │   │   └── results.py           # GET /results/{session_id}, GET /compare
│   ├── core/
│   │   ├── config.py                # Settings (env vars, Ollama config, etc.)
│   │   ├── security.py              # Optional API key/auth (no-auth for MVP)
│   │   ├── concurrency.py           # LlmLockManager for resource safety
│   │   ├── dependencies.py         # Shared: embeddings, llm, chroma client, db session
│   │   └── health.py                # Ollama health checks
│   ├── db/
│   │   ├── __init__.py
│   │   ├── database.py              # SQLModel engine (SQLite) and session management
│   │   ├── models.py                # SQLModel models: Document, QueryResult, Session
│   │   └── repositories.py          # Data access layer
│   ├── models/
│   │   ├── __init__.py
│   │   ├── enums.py                 # RAGTechnique enum + layers
│   │   ├── schemas.py               # Request/response Pydantic models
│   │   └── scoring.py               # Result scoring models and metrics
│   ├── rag/
│   │   ├── __init__.py
│   │   ├── pipeline.py              # LCEL Builder (LangChain Expression Language)
│   │   ├── limiter.py               # Atomic Context Limiter (drops full chunks)
│   │   ├── validators.py            # Three-layer validation system
│   │   └── techniques/              # HyDE, Fusion, CRAG implementations
│   │       ├── __init__.py
│   │       ├── basic.py
│   │       ├── hybrid_retrieval.py  # BM25 + Dense fusion (uses EnsembleRetriever)
│   │       ├── reranking.py
│   │       ├── hyde.py
│   │       ├── semantic_chunking.py # Uses langchain_experimental.SemanticChunker
│   │       ├── compression.py
│   │       ├── headers.py
│   │       ├── proposition.py
│   │       ├── self_rag.py
│   │       ├── crag.py
│   │       └── adaptive.py
│   ├── services/
│   │   ├── __init__.py
│   │   ├── ingestion.py             # PDF → text using LangChain PyMuPDFLoader + Chunkers + Progress Update
│   │   ├── vectorstore.py           # Chroma + LocalFileStore (Parent Document)
│   │   ├── bm25_manager.py          # BM25 rebuild from ChromaDB + LRU cache
│   │   ├── scoring.py               # Semantic Variance + inline metrics
│   │   └── tracing.py               # Capture retrieved chunks with sources
│   └── utils/
│       └── helpers.py               # Misc utilities
├── chromadb/                        # Vector Data (gitignored)
├── storage/
│   └── parents/                     # Parent Document Key-Value Store (LocalFileStore)
├── uploads/                         # Raw PDFs (gitignored)
├── playground.db                    # SQLite database file
├── .env                             # Environment variables
├── .env.example                     # Example environment variables
├── pyproject.toml                   # uv project config
├── uv.lock                          # uv lockfile
└── README.md
```

## Database Schema (SQLModel)

### Document Model

```python
class Document(SQLModel, table=True):
    id: UUID = Field(default_factory=uuid4, primary_key=True)
    session_id: UUID = Field(foreign_key="session.id")
    filename: str
    file_path: str  # Stored file location
    status: str  # pending, processing, completed, failed
    indexing_progress: int = Field(default=0)  # 0-100% for UI Progress Bars
    chunking_strategy: str  # standard, parent_document, semantic, proposition, headers
    chunk_size: int  # 256/512/1024/2048 - index-time parameter
    chunk_overlap: int = Field(default=0)  # index-time parameter
    chroma_collection: str  # ChromaDB collection name
    embedding_dimension: int = Field(default=768)  # Nomic default
    uploaded_at: datetime
    processed_at: datetime | None
```

### QueryResult Model (With Advanced Scoring)

```python
class QueryResult(SQLModel, table=True):
    id: UUID = Field(default_factory=uuid4, primary_key=True)
    session_id: UUID = Field(foreign_key="session.id")
    document_id: UUID = Field(foreign_key="document.id")
    query: str
    response: str  # Generated answer
    # Metrics (inline calculation)
    latency_ms: float
    token_count_est: int  # Estimate: len(prompt) / 4
    semantic_variance: float | None  # Score (0-1) comparing similarity to baseline result
    # JSON Blobs
    techniques_used: str  # JSON list of Layer 2 & 3 techniques
    query_params: str  # JSON: top_k, bm25_weight, temperature, etc.
    retrieved_chunks: str  # JSON with page/line metadata: text, page, line_start, line_end, score
    created_at: datetime
```

**Semantic Variance Scoring**: Calculates Cosine Similarity between two result answers to quantify "Did the strategy change the outcome?" Enables objective comparison of technique effectiveness.

### Session Model

- `id`: UUID (primary key)
- `created_at`: datetime
- `last_activity`: datetime

## Core API Endpoints

### Documents

- `POST /api/v1/documents/upload` - Upload PDF with **index-time configuration**
  - Body: `{file: File, chunk_size: int, chunk_overlap: int, chunking_strategy: str, session_id: UUID?}`
  - Returns: `{document_id, status: "processing", session_id}` (202 Accepted for slow techniques)
  - **Note**: These parameters are destructive - changing them requires re-upload
  - **Background Processing**: Slow techniques (Contextual Headers, Proposition) run in BackgroundTasks
  - **Status Polling**: UI polls `GET /documents/{id}` to check `status: processing → completed`

- `GET /api/v1/documents` - List documents (optionally filtered by session_id)
- `GET /api/v1/documents/{doc_id}` - Get document details
- `DELETE /api/v1/documents/{doc_id}` - Delete document, Chroma collection, and BM25 index

### RAG Queries

- `POST /api/v1/rag/query` - Execute RAG query with **query-time configuration**
  - Body: `{document_id: UUID, query: str, techniques: List[RAGTechnique], query_params: {top_k, bm25_weight, temperature}, session_id: UUID?}`
  - Returns: `{response, retrieved_chunks, scores, result_id}`
  - **Note**: Query-time parameters can be changed instantly without re-upload

### Results Management

- `GET /api/v1/results/{session_id}` - Get all results for session
- `GET /api/v1/results/compare?result_ids=id1,id2,...` - Compare multiple results side-by-side
- `GET /api/v1/results/{result_id}` - Get single result details

### Validation

- `POST /api/v1/rag/validate` - Validate technique combination (optional - for UI)
  - Body: `{techniques: List[RAGTechnique]}`
  - Returns: `{valid: bool, errors: List[str], warnings: List[str]}`

## Three-Layer Validation System

### Validation Rules (Critical for Backend)

**Layer 1 (Indexing Strategy) - Mutually Exclusive:**

- Must select exactly ONE: Standard, Semantic, Proposition, or Headers
- Cannot combine Semantic + Proposition
- Proposition cannot combine with Headers

**Layer 2 (Pipeline Components) - Mix & Match:**

- Must select EITHER Basic RAG OR Fusion Retrieval (not both)
- Reranking requires Basic or Fusion
- Compression requires Basic or Fusion
- HyDE is universally compatible (works with everything)

**Layer 3 (Advanced Controllers) - Max One:**

- Can select MAX ONE: Self-RAG, CRAG, or Adaptive
- Cannot combine Self-RAG + CRAG
- All Layer 3 techniques compatible with Layer 1 & 2

### Valid Combinations Examples

- Basic RAG: `[BASIC_RAG]` (Layer 2 only)
- Fusion + Reranking: `[FUSION_RETRIEVAL, RERANKING]` (Layer 2)
- HyDE + Fusion + Reranking: `[HYDE, FUSION_RETRIEVAL, RERANKING]` (Layer 2)
- Semantic Chunking (Layer 1) + Fusion + Reranking (Layer 2)
- Fusion + Reranking + Self-RAG: `[FUSION_RETRIEVAL, RERANKING, SELF_RAG]` (Layer 2 + Layer 3)
- CRAG: `[CRAG]` (Layer 3 standalone, but can use Layer 1 indexing)

### Invalid Combinations

- `[RERANKING]` alone (needs retrieval - Basic or Fusion)
- `[COMPRESSION]` alone (needs retrieval)
- `[BASIC_RAG, FUSION_RETRIEVAL]` (mutually exclusive - Fusion contains Basic)
- `[SEMANTIC_CHUNKING, PROPOSITION_CHUNKING]` (mutually exclusive - different storage)
- `[SELF_RAG, CRAG]` (mutually exclusive - logic collision)

## Index-Time vs Query-Time Parameters

### Critical UX Distinction

**Type A: Index-Time Parameters (Destructive - Require Re-Upload)**

- `chunk_size`: Size of text chunks
- `chunk_overlap`: Overlap between chunks
- `chunking_strategy`: Standard, Semantic, Proposition, Headers

**Type B: Query-Time Parameters (Non-Destructive - Instant)**

- `top_k`: Number of chunks to retrieve
- `bm25_weight`: Weight for BM25 in hybrid retrieval (0.0-1.0)
- `temperature`: LLM temperature
- `techniques`: Layer 2 & 3 techniques (HyDE, Reranking, etc.)

**UI Recommendation**: Separate into two sections:

- **"Indexing Strategy"** (forces re-process) - shown during upload
- **"Retrieval Strategy"** (instant run) - shown during query

## Result Scoring System

### Metrics to Track (Phase 1 - Simple)

- `latency_ms`: Query processing time (milliseconds)
- `token_count`: Total tokens used (input + output)
- `chunks_retrieved`: Number of chunks retrieved
- `response_length`: Generated response character count
- `chunk_relevance_score`: Average similarity score of retrieved chunks

### Future Metrics (Phase 2+)

- User feedback/ratings (thumbs up/down, 1-5 stars)
- Semantic similarity between query and response
- Retrieval precision/recall (if ground truth available)
- Cost per query (if provider pricing available)

### Scoring System (Inline Calculation + Semantic Variance)

**Basic Metrics** (calculated inline during request):
- `latency_ms`: Start/end timer around pipeline execution
- `token_count_est`: Estimate from prompt length (`len(prompt) / 4` approximation)
- `chunks_retrieved`: Count of retrieved chunks
- `response_length`: Character count of generated response
- `chunk_relevance_score`: Average similarity score from Chroma/BM25

**Advanced Metric: Semantic Variance** (for comparison):
- **Purpose**: Quantify "Did Strategy B produce a different answer than Strategy A?"
- **Calculation**: Cosine Similarity between two result embeddings
- **Implementation**: 
  - Use Ollama embeddings to embed both responses
  - Calculate cosine similarity: `cosine_sim(embedding_A, embedding_B)`
  - Score range: 0.0 (completely different) to 1.0 (identical)
- **Usage**: `GET /results/compare?result_ids=id1,id2` returns `semantic_variance: 0.73`
- **Interpretation**: 
  - High variance (>0.8): Strategies produced similar answers
  - Low variance (<0.5): Strategies produced meaningfully different answers

## Implementation Phases Overview

For detailed implementation steps, see the phase-specific documents:

- **Phase 0**: See [PHASE_0_FOUNDATION.md](PHASE_0_FOUNDATION.md) for project setup, dependencies, Ollama integration, database, ChromaDB, and core infrastructure
- **Phase 1**: See [PHASE_1_BACKEND.md](PHASE_1_BACKEND.md) for core RAG pipeline, document ingestion, validation system, query endpoints, and result management
- **Phase 2**: See [PHASE_2_ENHANCEMENTS.md](PHASE_2_ENHANCEMENTS.md) for HyDE, Semantic Chunking, Contextual Compression, Contextual Headers, and enhanced scoring
- **Phase 3**: See [PHASE_3_ADVANCED.md](PHASE_3_ADVANCED.md) for Self-RAG, Corrective RAG (CRAG), Adaptive Retrieval, and advanced flow controllers

## Key Implementation Details

### Simplified Pipeline Architecture (LCEL)

**Code Reduction Strategy**: Use LangChain LCEL (LangChain Expression Language v1.0.0+) instead of complex inheritance hierarchies.

**Implementation** (`app/rag/pipeline.py`):

```python
from langchain_core.runnables import RunnableParallel, RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

async def build_rag_chain(
    retriever, 
    llm, 
    use_reranker: bool = False, 
    use_hyde: bool = False,
    use_compression: bool = False,
    context_limiter: AtomicContextLimiter
):
    # 1. Expand Query (HyDE)
    if use_hyde:
        query_transformer = hyde_chain | (lambda x: x.text)
    else:
        query_transformer = RunnablePassthrough()

    # 2. Retrieve & Refine
    def retrieve_and_refine(q):
        docs = retriever.invoke(q)
        if use_reranker:
            docs = reranker_service.rerank(docs, q)
        if use_compression:
            docs = compressor.compress(docs, q)
        # Atomic limiter: drops full chunks, preserves sentence integrity
        return context_limiter.limit(docs)  # Prevent context overflow

    # 3. Final Generation Chain
    chain = (
        RunnableParallel({
            "context": query_transformer | retrieve_and_refine,
            "question": RunnablePassthrough()
        })
        | prompt_template
        | llm
        | StrOutputParser()
    )
    
    return chain
```

**Benefits**:
- Reduces `rag/pipeline.py` complexity by ~60%
- Declarative chain composition
- Easier to test and debug
- Better alignment with LangChain v1.0.0+ best practices
- Type-safe with Pydantic v2

### Document Processing Workflow

1. **Upload Phase (Index-Time Parameters)**:

   - User uploads PDF with `chunk_size`, `chunk_overlap`, `chunking_strategy`
   - Extract text using LangChain PyMuPDFLoader
   - Apply chunking strategy:
     - Standard: RecursiveCharacterTextSplitter
     - Semantic: SemanticChunker with Ollama embeddings
     - Proposition: LLM-based atomic fact extraction
     - Headers: Standard/Semantic + LLM-generated titles
   - Create ChromaDB collection with explicit dimension (768)
   - Generate embeddings using Ollama Nomic
   - Store chunks in ChromaDB
   - Build BM25 index (no persistence - will rebuild on-demand from ChromaDB)
   - Store document metadata in SQLite

2. **Query Phase (Query-Time Parameters)**:

   - User submits query with techniques and query parameters
   - Validate technique combination (three-layer system)
   - Execute pipeline:
     - HyDE (if selected): Expand query
     - Retrieval: Basic RAG or Fusion (rebuild BM25 from ChromaDB if Fusion)
     - Reranking (if selected): Re-order results
     - Compression (if selected): Compress context
     - **Context Limiting**: Trim chunks to fit LLM context window (drop lowest-scored if needed)
     - Generation: LLM generates answer
   - Calculate scores inline (latency, tokens, chunks, relevance)
   - Store result in SQLite with scores
   - Return response with source chunks

### BM25 Rebuild Strategy (No Persistence)

**Simplified Approach - Rebuild from ChromaDB**:

1. **During Document Processing**:

   - Build BM25 index from chunked text (for validation)
   - **No persistence** - chunks are already in ChromaDB
   - Index will be rebuilt on-demand when needed

2. **During Hybrid Retrieval**:

   - Check in-memory LRU cache first
   - If not cached: Load all chunks from ChromaDB collection
   - Rebuild BM25 index: `BM25Okapi([chunk.text for chunk in chunks])`
   - Cache in memory (survives until server restart)
   - Use with LangChain EnsembleRetriever

3. **Performance**:

   - Rebuild time: <1 second for 1000 chunks (just tokenization + sparse matrix)
   - LRU cache prevents repeated rebuilds within session
   - No versioning complexity (each document has one ChromaDB collection)

4. **Benefits**:

   - No pickle files to manage
   - No race conditions
   - No versioning complexity
   - Simpler code
   - Fast enough for MVP

5. **Future Optimization** (if needed):

   - Add persistence only if documents have 5000+ chunks (rebuild >5s)
   - Or if users query same document 10+ times in session

### Session Management

- Session ID auto-generated if not provided in request
- All documents and results linked to session
- Enables result comparison within session
- Simple anonymous sessions (no auth required for MVP)
- Session cleanup (optional - for abandoned sessions)

### Async Patterns

- All FastAPI endpoints async
- **Background Tasks**: Use FastAPI BackgroundTasks for slow indexing (Contextual Headers, Proposition)
- Async document processing with status polling for slow techniques
- Async LLM calls via LangChain Ollama
- Async database operations with SQLModel
- Async Chroma operations

### Atomic Context Limiter

**Component**: `app.rag.limiter.AtomicContextLimiter`

**Problem**: Hard-cutting text at character limits risks broken sentences and context loss.

**Solution**: Drop full chunks instead of slicing:
1. Determines available budget: `Total - (SystemPrompt + Query)`
2. Adds full chunks until budget is near full
3. **Drops** the lowest-ranked chunk entirely rather than slicing it
4. Preserves sentence integrity and context coherence

**Implementation**:
```python
class AtomicContextLimiter:
    def __init__(self, max_tokens: int, token_estimator):
        self.max_tokens = max_tokens
        self.estimate = token_estimator
    
    def limit(self, chunks: List[Document], query: str) -> List[Document]:
        system_prompt_tokens = self.estimate(SYSTEM_PROMPT)
        query_tokens = self.estimate(query)
        available = self.max_tokens - system_prompt_tokens - query_tokens - 500  # Buffer
        
        selected = []
        total = 0
        for chunk in sorted(chunks, key=lambda x: x.metadata['score'], reverse=True):
            chunk_tokens = self.estimate(chunk.page_content)
            if total + chunk_tokens <= available:
                selected.append(chunk)
                total += chunk_tokens
            else:
                break  # Drop this chunk entirely (don't slice)
        return selected
```

### Error Handling Strategy

- **PDF parsing**: catch PyMuPDFLoader errors, return 400 with descriptive message
- **Ollama API**: health check at startup, retry logic (3 attempts), timeout handling (30s), clear setup instructions in errors
- **Chroma**: connection retry, collection not found handling, graceful degradation
- **Invalid techniques**: 400 with detailed validation error (which techniques conflict, what's missing, layer violations)
- **Large files**: size limit validation (50MB max), return 400 with clear message
- **SQLite**: connection retry, file permission errors
- **BM25 rebuild**: handle ChromaDB collection errors, rebuild index gracefully

### File Storage

- Store uploaded PDFs in `uploads/{document_id}/` directory (filesystem)
- File naming: `{document_id}_{filename}`
- **No BM25 pickle files** - indices rebuilt from ChromaDB on-demand
- Easy migration to S3 later if needed
- Cleanup on document deletion (PDF + Chroma collection)

### Chroma Persistence

- Chroma data stored in `chromadb/` directory
- Uses PersistentClient (embedded mode, no server)
- Persists across restarts
- One collection per document: `doc_{document_id}`
- Explicit embedding dimension: 768 (for Nomic)
- Collection cleanup on document deletion

## Dependencies

Add to `pyproject.toml`:

- `fastapi` - Web framework
- `uvicorn[standard]` - ASGI server
- `pydantic` - Data validation
- `sqlmodel` - Database ORM
- `langchain>=1.0.0` - RAG framework (CRITICAL: v1.0.0+ required)
- `langchain-core` - Core LangChain components
- `langchain-community` - LangChain community integrations (PyMuPDFLoader, LocalFileStore)
- `langchain-experimental` - SemanticChunker, advanced techniques
- `langchain-ollama` - Ollama LLM and embeddings integration
- `langchain-chroma` - ChromaDB integration
- `langchain-text-splitters` - Chunking strategies
- `chromadb` - Vector store
- `rank-bm25` - BM25 implementation
- `sentence-transformers` - Cross-encoder models
- `scikit-learn` - For semantic variance (cosine similarity)
- `tavily-python` - Web search for CRAG (recommended)
- `duckduckgo-search` - Web search fallback
- `python-multipart` - File uploads for FastAPI
- `python-dotenv` - Environment variable management

**Note**: No PostgreSQL drivers needed (using SQLite)

## Environment Variables

Required in `.env`:

```
# Database
DATABASE_URL=sqlite:///playground.db

# Ollama
OLLAMA_BASE_URL=http://localhost:11434
OLLAMA_LLM_MODEL=llama3.2  # or your preferred local LLM
OLLAMA_EMBEDDING_MODEL=nomic-embed-text

# Embeddings
EMBEDDING_DIMENSION=768  # Nomic v1.5 default

# Chroma
CHROMA_PERSIST_DIR=./chromadb

# File Upload
MAX_FILE_SIZE_MB=50
UPLOAD_DIR=./uploads

# Web Search (for CRAG)
TAVILY_API_KEY=your_tavily_key_here  # Optional, recommended
USE_TAVILY=true  # Set to false to use DuckDuckGo

# Optional
LOG_LEVEL=INFO
```

## Setup Requirements

### Prerequisites

1. **Ollama Installation**:
   ```bash
   # Install Ollama (see https://ollama.ai)
   # Pull required models:
   ollama pull nomic-embed-text
   ollama pull llama3.2  # or your preferred LLM
   ```

2. **Start Ollama Service**:
   ```bash
   ollama serve
   # Or ensure it's running as a service
   ```

3. **Python Environment**:
   ```bash
   uv sync  # Install all dependencies
   ```


### Health Check on Startup

Application will verify:

- Ollama API accessible at `http://localhost:11434`
- `nomic-embed-text` model available
- If checks fail, application exits with clear error message and setup instructions

## Open Questions Resolved

✅ LLM Provider: LangChain Ollama (local execution)

✅ Database: SQLite with SQLModel (zero-setup)

✅ Embeddings: Ollama Nomic Embeddings (local, 768 dimensions)

✅ Chunk Size: Per-document index-time parameter

✅ Chunking Strategy: Index-time parameter (re-upload to change)

✅ Web Search: Tavily (recommended) or DuckDuckGo (fallback)

✅ Streaming: Not required

✅ Result Management: Session-based with scoring

✅ File Storage: Filesystem for MVP

✅ Session Management: Anonymous sessions, auto-generated IDs

✅ BM25 Strategy: Rebuild from ChromaDB chunks on-demand (with LRU cache)
✅ LangChain Version: v1.0.0+ (CRITICAL for modern LCEL patterns)
✅ Parent Document Retrieval: Dual storage (ChromaDB + LocalFileStore)
✅ LlmLockManager: Concurrency control to prevent resource exhaustion
✅ Atomic Context Limiter: Drops full chunks instead of slicing
✅ Semantic Variance Scoring: Cosine similarity for result comparison
✅ Indexing Progress: 0-100% progress tracking for UI

✅ PDF Loader: LangChain PyMuPDFLoader

✅ ChromaDB: PersistentClient (embedded mode)

✅ Validation: Three-layer system (Indexing, Pipeline, Advanced)

✅ Parameter Separation: Index-time vs Query-time distinction

## Performance Considerations

### Background Task Strategy

**Problem**: Slow indexing techniques (Contextual Headers, Proposition Chunking) can take 30+ minutes, causing HTTP timeouts.

**Solution**: Use FastAPI BackgroundTasks

1. **Upload Endpoint**:
   - Return `202 Accepted` immediately for slow techniques
   - Set document status to `processing`
   - Launch background task for indexing

2. **Background Task**:
   - Process document asynchronously
   - Update status: `processing → completed` (or `failed` on error)
   - Store results in ChromaDB and SQLite

3. **UI Polling**:
   - UI polls `GET /documents/{id}` to check status
   - Show progress indicator while `status: processing`
   - Display results when `status: completed`

**Implementation**:
```python
from fastapi import BackgroundTasks

@router.post("/upload")
async def upload_document(
    file: UploadFile,
    background_tasks: BackgroundTasks,
    ...
):
    doc = create_document(...)
    
    if is_slow_technique(chunking_strategy):
        background_tasks.add_task(process_document_slow, doc.id)
        return {"status": "processing", "document_id": doc.id}  # 202
    else:
        await process_document_fast(doc.id)
        return {"status": "completed", "document_id": doc.id}
```

### Slow Techniques (Require UI Warnings)

- **Contextual Headers**: ~30 minutes for 20-page PDF on local LLM
- **Proposition Chunking**: Extremely slow (LLM rewrites every sentence)
- **Semantic Chunking**: Slower than standard (embeds during split)

### Memory Requirements

- **Ollama**: 4GB+ RAM (depending on LLM model)
- **Cross-Encoder (Reranking)**: 1GB+ RAM
- **Total**: ~5-6GB RAM recommended for full functionality

### Optimization Opportunities

- Cache embeddings for semantic chunking (if same document re-processed)
- **Background Tasks**: Implemented for slow indexing techniques (prevents timeouts)
- Progress indicators for long-running operations
- **Context Window Limiting**: Implemented to prevent overflow
- **Pipeline Simplification**: Using LCEL reduces code complexity by ~60%
- **Scoring Simplification**: Inline calculation eliminates service class overhead

## 9. Success Criteria

Phase 1 MVP is complete when:

1. **Stability**: Application handles "Indexing while Querying" gracefully (via LlmLockManager) without crashing
2. **Traceability**: Every answer includes exact Page/Line numbers from the PDF
3. **Comparison**: Users can prove that "Strategy B" is better than "Strategy A" using Semantic Variance score
4. **Completeness**: Implements at least one technique from each of the 3 Layers successfully
5. **Functionality**:
   - Users can upload PDFs with index-time configuration (Standard + Parent Document)
   - Users can query documents with query-time technique selection
   - Three-layer validation system prevents invalid combinations
   - Results are stored and can be compared side-by-side with semantic variance
   - At least 3 foundational techniques work (Basic, Fusion, Reranking)
6. **Infrastructure**:
   - Ollama health checks work correctly
   - BM25 indices rebuild correctly from ChromaDB (with LRU cache)
   - Background tasks work for slow indexing techniques
   - Atomic context limiter prevents overflow without breaking sentences
   - Indexing progress tracking works (0-100%)
   - Error handling is robust
   - API is documented and testable
   - Index-time vs query-time parameters are clearly separated