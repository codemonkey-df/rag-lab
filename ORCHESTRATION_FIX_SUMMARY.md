# Orchestration Pipeline Fix Summary

## Problems Fixed

1. **SelfRAG and CRAG Parameter Mismatch**: These classes expected a `BaseRetriever` instance but were receiving raw parameters like `document_id`.

2. **Missing Retriever Creation**: The orchestration pipeline was not creating retriever instances before passing them to SelfRAG/CRAG.

3. **Structured Output Not Supported**: `OllamaLLM` does not support `with_structured_output()`, causing `NotImplementedError` for all orchestration techniques using structured outputs.

## Solutions Implemented

### 1. Created Structured Output Fallback Utility (`app/rag/utils.py`)

**New File**: `app/rag/utils.py`

- **`build_structured_chain_with_fallback()`**: Async function that attempts native structured output first, falls back to JSON parsing
- **`build_structured_chain()`**: Synchronous wrapper with same fallback logic
- **Features**:
  - Tries `llm.with_structured_output()` first
  - On `NotImplementedError`, generates JSON schema and asks LLM for JSON output
  - Parses and validates JSON response against Pydantic model
  - Handles both synchronous and asynchronous invocation

### 2. Created Retriever Adapter (`app/rag/techniques/retrieval/basic.py`)

**New Class**: `BaseRetrievalAdapter`

- Wraps `BaseRetrieval` instances to expose them as `BaseRetriever`
- Implements `_aget_relevant_documents()` for async retrieval
- Provides bridge between different retriever interfaces
- Includes error handling with fallback to empty results

### 3. Fixed Orchestration Pipeline (`app/rag/pipelines/orchestration_pipeline.py`)

**Key Changes**:
- For `SelfRAG` and `CRAG`:
  - Create `BaseRetrieval` instance using `RetrievalFactory`
  - Wrap it with `BaseRetrievalAdapter` to convert to `BaseRetriever`
  - Pass retriever to orchestration technique
- For `AdaptiveRetrieval`:
  - Keep current parameter passing (already handles `document_id`)
- Improved logging for retriever creation

### 4. Updated Orchestration Techniques

**SelfRAG** (`app/rag/techniques/orchestration/self_rag.py`):
- Updated `_build_structured_chain()` to use `build_structured_chain_with_fallback()`
- Removed direct `with_structured_output()` call

**CRAG** (`app/rag/techniques/orchestration/crag.py`):
- Updated `_build_structured_chain()` to use `build_structured_chain_with_fallback()`
- Removed direct `with_structured_output()` call

**AdaptiveRetrieval** (`app/rag/techniques/orchestration/adaptive.py`):
- Updated both `BaseRetrievalStrategy._build_structured_chain()` and `AdaptiveRetrieval._build_structured_chain()`
- Both now use `build_structured_chain_with_fallback()`

## Files Modified

1. ✅ `app/rag/utils.py` - Created
2. ✅ `app/rag/pipelines/orchestration_pipeline.py` - Updated
3. ✅ `app/rag/techniques/retrieval/basic.py` - Added `BaseRetrievalAdapter`
4. ✅ `app/rag/techniques/orchestration/self_rag.py` - Updated imports and `_build_structured_chain()`
5. ✅ `app/rag/techniques/orchestration/crag.py` - Updated imports and `_build_structured_chain()`
6. ✅ `app/rag/techniques/orchestration/adaptive.py` - Updated imports and both `_build_structured_chain()` methods

## Testing

All modules:
- ✅ Import successfully
- ✅ Compile without syntax errors
- ✅ Pass linter checks
- ✅ OrchestrationPipeline instantiates correctly

## Error Cases Handled

1. **Native structured output unavailable**: Falls back to JSON parsing
2. **JSON parsing fails**: Returns appropriate error with logging
3. **Retriever creation fails**: Returns EmptyRetriever to gracefully degrade
4. **Async retrieval fails in adapter**: Returns empty list and logs error

## Backward Compatibility

- All changes are internal to implementation
- Public APIs remain unchanged
- No breaking changes to existing code
