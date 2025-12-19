# Orchestration Pipeline Parameter Mismatch Fix

## Summary
Fixed parameter mismatches in the orchestration pipeline that were causing `TypeError` exceptions when calling different orchestration techniques.

## Issues Fixed

### 1. SelfRAG Parameter Mismatch (Lines 723, 784, 815, 840)
**Error**: `SelfRAG.process() got an unexpected keyword argument 'temperature'`

**Root Cause**: `SelfRAG.process()` only accepts `question` and `top_k` parameters, but the pipeline was passing `temperature` and `user_context`.

**Fix**: Now conditionally passes only required parameters to SelfRAG:
```python
result = await orchestrator.process(
    question=query,
    top_k=top_k,
)
```

### 2. CRAG Parameter Mismatch (Lines 845, 906, 918, 962)
**Error**: `CRAG.process() got an unexpected keyword argument 'temperature'`

**Root Cause**: `CRAG.process()` only accepts `question` and `top_k` parameters, but the pipeline was passing `temperature` and `user_context`.

**Fix**: Now conditionally passes only required parameters to CRAG (same as SelfRAG).

### 3. AdaptiveRetrieval Parameter Mismatch (Lines 966, 997, 1022)
**Error**: `AdaptiveRetrieval.process() got an unexpected keyword argument 'question'`

**Root Cause**: `AdaptiveRetrieval.process()` expects `query` (not `question`) and accepts `temperature` and `user_context`, but the pipeline was passing `question` instead of `query`.

**Fix**: Now conditionally passes correct parameters to AdaptiveRetrieval:
```python
result = await orchestrator.process(
    query=query,
    top_k=top_k,
    temperature=kwargs.get("temperature", 0.7),
    user_context=kwargs.get("user_context"),
)
```

## Implementation Details

**File Modified**: `app/rag/pipelines/orchestration_pipeline.py` (Lines 102-116)

**Change Type**: Added technique-specific conditional logic to handle different orchestrator signatures.

```python
# Execute orchestration pipeline with technique-specific parameters
if technique == RAGTechnique.ADAPTIVE_RETRIEVAL:
    # AdaptiveRetrieval uses 'query' not 'question', and accepts temperature/user_context
    result = await orchestrator.process(
        query=query,
        top_k=top_k,
        temperature=kwargs.get("temperature", 0.7),
        user_context=kwargs.get("user_context"),
    )
else:
    # SelfRAG and CRAG only accept 'question' and 'top_k'
    result = await orchestrator.process(
        question=query,
        top_k=top_k,
    )
```

## Method Signatures (Reference)

- **SelfRAG.process()**: `async def process(self, question: str, top_k: int = 5) -> Dict[str, Any]`
- **CRAG.process()**: `async def process(self, question: str, top_k: int = 5) -> Dict[str, Any]`
- **AdaptiveRetrieval.process()**: `async def process(self, query: str, top_k: Optional[int] = None, bm25_weight: Optional[float] = None, temperature: float = 0.7, user_context: Optional[str] = None) -> Dict[str, Any]`

## Testing

All error cases from the test logs are now addressed:
- ✓ SELF_RAG no longer receives unexpected 'temperature' argument
- ✓ CRAG no longer receives unexpected 'temperature' argument
- ✓ ADAPTIVE_RETRIEVAL now receives 'query' instead of 'question'
- ✓ ADAPTIVE_RETRIEVAL properly receives 'temperature' and 'user_context'

## Files Changed
1. `app/rag/pipelines/orchestration_pipeline.py` - Modified execute() method (Lines 102-116)

## No Breaking Changes
- The fix is backward compatible
- All orchestration techniques continue to work as expected
- Parameter handling is transparent to callers
