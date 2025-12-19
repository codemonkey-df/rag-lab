"""
Pipeline stages using Chain of Responsibility pattern.

Each stage processes the pipeline context and passes it to the next stage.
This allows flexible composition of pipeline steps.
"""

import logging
from abc import ABC, abstractmethod
from typing import Any, Dict, List
from uuid import UUID

logger = logging.getLogger(__name__)


class PipelineContext:
    """Context object that flows through the pipeline."""

    def __init__(
        self,
        query: str,
        document_id: UUID,
        top_k: int = 5,
        temperature: float = 0.7,
        **kwargs,
    ):
        """
        Initialize pipeline context.

        Args:
            query: Original user query
            document_id: UUID of document to query
            top_k: Number of documents to retrieve
            temperature: LLM temperature
            **kwargs: Additional context
        """
        self.query = query
        self.original_query = query  # Store original for reranking
        self.document_id = document_id
        self.top_k = top_k
        self.temperature = temperature
        self.documents = []
        self.answer = ""
        self.metadata = {}
        self.extra = {**kwargs}

    def to_dict(self) -> Dict[str, Any]:
        """Convert context to dictionary."""
        return {
            "answer": self.answer,
            "documents": self.documents,
            "metadata": self.metadata,
        }


class PipelineStage(ABC):
    """Abstract base class for pipeline stages."""

    @abstractmethod
    async def process(self, context: PipelineContext) -> PipelineContext:
        """
        Process pipeline context.

        Args:
            context: Current pipeline context

        Returns:
            Updated pipeline context
        """
        pass


class QueryExpansionStage(PipelineStage):
    """Stage that expands queries using query expansion techniques."""

    def __init__(self, expansion_technique):
        """
        Initialize query expansion stage.

        Args:
            expansion_technique: BaseQueryExpansion instance
        """
        self.expansion_technique = expansion_technique

    async def process(self, context: PipelineContext) -> PipelineContext:
        """
        Expand the query.

        Args:
            context: Pipeline context

        Returns:
            Updated context with expanded query
        """
        try:
            expanded = await self.expansion_technique.expand(context.query)
            context.query = expanded
            logger.info(f"Query expanded: {len(context.original_query)} -> {len(context.query)} chars")
        except Exception as e:
            logger.warning(f"Query expansion failed, using original query: {e}")
            # Fallback to original query
        return context


class RetrievalStage(PipelineStage):
    """Stage that retrieves documents."""

    def __init__(self, retrieval_technique):
        """
        Initialize retrieval stage.

        Args:
            retrieval_technique: BaseRetrieval instance
        """
        self.retrieval_technique = retrieval_technique

    async def process(self, context: PipelineContext) -> PipelineContext:
        """
        Retrieve documents.

        Args:
            context: Pipeline context

        Returns:
            Updated context with retrieved documents
        """
        try:
            docs = await self.retrieval_technique.retrieve(
                query=context.query,
                document_id=context.document_id,
                top_k=context.top_k,
            )
            context.documents = docs
            logger.info(f"Retrieved {len(docs)} documents")
        except Exception as e:
            logger.error(f"Retrieval failed: {e}", exc_info=True)
            context.documents = []
        return context


class FilteringStage(PipelineStage):
    """Stage that filters/reranks documents."""

    def __init__(self, filters: List):
        """
        Initialize filtering stage with multiple filters.

        Args:
            filters: List of BaseFiltering instances
        """
        self.filters = filters

    async def process(self, context: PipelineContext) -> PipelineContext:
        """
        Apply filters in sequence.

        Args:
            context: Pipeline context

        Returns:
            Updated context with filtered documents
        """
        docs = context.documents
        for filter_technique in self.filters:
            try:
                docs = await filter_technique.filter(
                    documents=docs,
                    query=context.original_query,  # Use original query for filtering
                    top_k=context.top_k,
                )
                logger.info(
                    f"{filter_technique.get_name()} filter reduced docs to {len(docs)}"
                )
            except Exception as e:
                logger.warning(f"Filtering failed: {e}, continuing with original docs")
                # Continue with original docs
        context.documents = docs
        return context


class ContextLimitingStage(PipelineStage):
    """Stage that limits context to fit in LLM context window."""

    def __init__(self, limiter):
        """
        Initialize context limiting stage.

        Args:
            limiter: Context limiter instance
        """
        self.limiter = limiter

    async def process(self, context: PipelineContext) -> PipelineContext:
        """
        Limit context.

        Args:
            context: Pipeline context

        Returns:
            Updated context with limited documents
        """
        try:
            # Combine context with page numbers
            context_parts = []
            for doc in context.documents:
                try:
                    page = (
                        doc.metadata.get("page", "?")
                        if hasattr(doc, "metadata")
                        else "?"
                    )
                    page_content = (
                        doc.page_content if hasattr(doc, "page_content") else str(doc)
                    )
                    context_parts.append(f"[Page {page}] {page_content}")
                except Exception as e:
                    logger.warning(f"Error processing document for context: {e}")
                    continue

            combined_context = "\n\n".join(context_parts)
            context.metadata["combined_context"] = combined_context
            logger.info(f"Combined context: {len(combined_context)} chars")
        except Exception as e:
            logger.warning(f"Context limiting failed: {e}")
            context.metadata["combined_context"] = ""
        return context


class GenerationStage(PipelineStage):
    """Stage that generates the final answer."""

    def __init__(self, llm, prompt_template):
        """
        Initialize generation stage.

        Args:
            llm: Language model
            prompt_template: Prompt template for generation
        """
        self.llm = llm
        self.prompt_template = prompt_template

    async def process(self, context: PipelineContext) -> PipelineContext:
        """
        Generate answer.

        Args:
            context: Pipeline context

        Returns:
            Updated context with answer
        """
        try:
            from langchain_core.output_parsers import StrOutputParser

            # Get combined context
            combined_context = context.metadata.get("combined_context", "")
            if not combined_context:
                logger.warning("No context available for generation")
                context.answer = (
                    "I couldn't retrieve any relevant information to answer your question."
                )
                return context

            # Generate answer
            result = await self.prompt_template.ainvoke(
                {"context": combined_context, "question": context.original_query}
            )
            answer = await self.llm.ainvoke(result.messages)
            context.answer = StrOutputParser().invoke(answer) or ""

            logger.info(f"Generated answer: {len(context.answer)} chars")
        except Exception as e:
            logger.error(f"Generation failed: {e}", exc_info=True)
            context.answer = "I encountered an error while generating a response. Please try again."
        return context
