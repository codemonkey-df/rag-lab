"""
Atomic context limiter - drops full chunks instead of slicing
"""
from typing import List
from langchain_core.documents import Document
from app.core.config import get_settings


class AtomicContextLimiter:
    """
    Limits context by dropping full chunks instead of slicing.
    
    Preserves sentence integrity and context coherence.
    """
    
    def __init__(self, max_tokens: int | None = None):
        settings = get_settings()
        self.max_tokens = max_tokens or settings.max_context_tokens
    
    def _estimate_tokens(self, text: str) -> int:
        """Estimate token count (rough: 1 token ≈ 4 characters)."""
        return len(text) // 4
    
    def limit(
        self, 
        chunks: List[Document], 
        query: str,
        system_prompt: str = ""
    ) -> List[Document]:
        """
        Limit chunks to fit within token budget.
        
        Strategy:
        1. Calculate available budget: Total - (SystemPrompt + Query + Buffer)
        2. Add full chunks until budget is near full
        3. Drop lowest-ranked chunk entirely (don't slice)
        
        Args:
            chunks: List of Document objects (should be sorted by relevance)
            query: User query text
            system_prompt: System prompt text
        
        Returns:
            Filtered list of chunks that fit within budget
        """
        system_tokens = self._estimate_tokens(system_prompt)
        query_tokens = self._estimate_tokens(query)
        buffer = 500  # Safety buffer
        
        available = self.max_tokens - system_tokens - query_tokens - buffer
        
        if available <= 0:
            return []  # No space available
        
        selected = []
        total_tokens = 0
        
        # Chunks should already be sorted by relevance (highest first)
        for chunk in chunks:
            chunk_tokens = self._estimate_tokens(chunk.page_content)
            
            if total_tokens + chunk_tokens <= available:
                selected.append(chunk)
                total_tokens += chunk_tokens
            else:
                # Drop this chunk entirely (don't slice)
                break
        
        return selected
