"""
Standard RAG Pipeline using Template Method pattern.

Defines the common workflow for standard RAG queries:
1. Query Expansion (optional)
2. Retrieval
3. Filtering (reranking, compression)
4. Context Limiting
5. Generation
"""

import logging
from typing import Any, Dict, List, Optional
from uuid import UUID

from langchain_core.prompts import ChatPromptTemplate

from app.core.dependencies import get_llm
from app.models.enums import RAGTechnique
from app.rag.limiter import AtomicContextLimiter
from app.rag.pipelines.stages import (
    ContextLimitingStage,
    FilteringStage,
    GenerationStage,
    PipelineContext,
    PipelineStage,
    QueryExpansionStage,
    RetrievalStage,
)
from app.rag.techniques.filtering.factory import FilteringFactory
from app.rag.techniques.query_expansion.factory import QueryExpansionFactory
from app.rag.techniques.retrieval.factory import RetrievalFactory

logger = logging.getLogger(__name__)

SYSTEM_PROMPT = """You are a helpful assistant that answers questions based on the provided context.
Use only the information from the context to answer. If the context doesn't contain enough information, say so.
Always cite page numbers when referencing specific information.

Important: Respond in the same language as the user's question."""


class StandardRAGPipeline:
    """
    Standard RAG Pipeline using template method pattern.

    Orchestrates the execution of pipeline stages to process a user query
    and generate an answer.
    """

    def __init__(self):
        """Initialize standard RAG pipeline."""
        self.stages: List[PipelineStage] = []

    def build_stages(
        self,
        techniques: List[RAGTechnique],
        document_id: UUID,
        top_k: int = 5,
        bm25_weight: float = 0.5,
        temperature: float = 0.7,
        chunking_strategy: Optional[str] = None,
        chunk_size: Optional[int] = None,
        chunk_overlap: Optional[int] = None,
    ) -> "StandardRAGPipeline":
        """
        Build pipeline stages based on selected techniques.

        This is the Template Method hook for customizing pipeline construction.

        Args:
            techniques: List of RAGTechnique values to apply
            document_id: UUID of document to query
            top_k: Number of documents to retrieve
            bm25_weight: Weight for BM25 in hybrid retrieval
            temperature: LLM temperature
            chunking_strategy: Optional chunking strategy
            chunk_size: Optional chunk size
            chunk_overlap: Optional chunk overlap

        Returns:
            Self for method chaining
        """
        self.stages = []

        # Stage 1: Query Expansion (optional)
        if RAGTechnique.HYDE in techniques:
            try:
                expansion = QueryExpansionFactory.create(RAGTechnique.HYDE)
                self.stages.append(QueryExpansionStage(expansion))
                logger.info("Added QueryExpansion stage (HyDE)")
            except Exception as e:
                logger.warning(f"Failed to add QueryExpansion stage: {e}")

        # Stage 2: Retrieval (required)
        retrieval_technique = (
            RAGTechnique.FUSION_RETRIEVAL
            if RAGTechnique.FUSION_RETRIEVAL in techniques
            else RAGTechnique.BASIC_RAG
        )

        try:
            retriever = RetrievalFactory.create(
                retrieval_technique,
                document_id=document_id,
                top_k=top_k,
                bm25_weight=bm25_weight,
                chunking_strategy=chunking_strategy,
                chunk_size=chunk_size,
                chunk_overlap=chunk_overlap,
            )
            self.stages.append(RetrievalStage(retriever))
            logger.info(f"Added Retrieval stage ({retrieval_technique})")
        except Exception as e:
            logger.error(f"Failed to add Retrieval stage: {e}")
            raise

        # Stage 3: Filtering (optional)
        filters = []
        if RAGTechnique.RERANKING in techniques:
            try:
                reranker = FilteringFactory.create(RAGTechnique.RERANKING)
                filters.append(reranker)
                logger.info("Added Reranking filter")
            except Exception as e:
                logger.warning(f"Failed to add Reranking filter: {e}")

        if RAGTechnique.CONTEXTUAL_COMPRESSION in techniques:
            try:
                compressor = FilteringFactory.create(
                    RAGTechnique.CONTEXTUAL_COMPRESSION
                )
                filters.append(compressor)
                logger.info("Added Compression filter")
            except Exception as e:
                logger.warning(f"Failed to add Compression filter: {e}")

        if filters:
            self.stages.append(FilteringStage(filters))

        # Stage 4: Context Limiting
        limiter = AtomicContextLimiter()
        self.stages.append(ContextLimitingStage(limiter))
        logger.info("Added ContextLimiting stage")

        # Stage 5: Generation
        llm = get_llm()
        llm.temperature = temperature
        prompt = ChatPromptTemplate.from_messages(
            [
                ("system", SYSTEM_PROMPT),
                ("human", "Context: {context}\n\nQuestion: {question}"),
            ]
        )
        self.stages.append(GenerationStage(llm, prompt))
        logger.info("Added Generation stage")

        return self

    async def execute(self, query: str, document_id: UUID, **kwargs) -> Dict[str, Any]:
        """
        Execute the pipeline (Template Method).

        Args:
            query: User query
            document_id: UUID of document to query
            **kwargs: Additional parameters (top_k, temperature, etc.)

        Returns:
            Dictionary containing answer and metadata
        """
        # Create context
        context = PipelineContext(
            query=query,
            document_id=document_id,
            **kwargs,
        )

        # Execute stages in sequence (Chain of Responsibility)
        for stage in self.stages:
            context = await stage.process(context)

        return context.to_dict()
