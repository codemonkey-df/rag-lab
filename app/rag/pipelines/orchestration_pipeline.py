"""
Orchestration Pipeline wrapper.

Orchestration techniques handle their own complete RAG pipeline,
so this pipeline simply delegates to the technique's process method.
"""

import logging
from typing import Any, Dict
from uuid import UUID

from app.models.enums import RAGTechnique
from app.rag.techniques.orchestration.factory import OrchestrationFactory
from app.rag.techniques.retrieval.basic import BaseRetrievalAdapter
from app.rag.techniques.retrieval.factory import RetrievalFactory

logger = logging.getLogger(__name__)


class OrchestrationPipeline:
    """
    Wrapper for orchestration techniques.

    Orchestration techniques (Self-RAG, CRAG, Adaptive) manage their own pipelines,
    so this wrapper simply creates and delegates to the appropriate technique.
    """

    async def execute(
        self,
        query: str,
        document_id: UUID,
        technique: RAGTechnique,
        **kwargs,
    ) -> Dict[str, Any]:
        """
        Execute orchestration technique pipeline.

        Args:
            query: User query
            document_id: UUID of document to query
            technique: Which orchestration technique to use
            **kwargs: Additional parameters (top_k, temperature, etc.)

        Returns:
            Dictionary containing answer and metadata
        """
        try:
            top_k = kwargs.get("top_k", 5)
            bm25_weight = kwargs.get("bm25_weight", 0.5)
            chunking_strategy = kwargs.get("chunking_strategy")
            chunk_size = kwargs.get("chunk_size")
            chunk_overlap = kwargs.get("chunk_overlap")

            # For SelfRAG and CRAG, we need to create a retriever first
            if technique in (RAGTechnique.SELF_RAG, RAGTechnique.CRAG):
                # Determine retrieval technique (default to BASIC_RAG)
                retrieval_technique = kwargs.get(
                    "retrieval_technique", RAGTechnique.BASIC_RAG
                )
                if retrieval_technique not in (
                    RAGTechnique.BASIC_RAG,
                    RAGTechnique.FUSION_RETRIEVAL,
                ):
                    retrieval_technique = RAGTechnique.BASIC_RAG

                logger.info(
                    f"Creating retriever ({retrieval_technique}) for {technique}"
                )

                # Create the BaseRetrieval instance
                base_retrieval = RetrievalFactory.create(
                    retrieval_technique,
                    document_id=document_id,
                    top_k=top_k,
                    bm25_weight=bm25_weight,
                    chunking_strategy=chunking_strategy,
                    chunk_size=chunk_size,
                    chunk_overlap=chunk_overlap,
                )

                # Wrap it as BaseRetriever using adapter
                retriever = BaseRetrievalAdapter(
                    base_retrieval, document_id=document_id, top_k=top_k
                )

                # Create orchestration instance with retriever
                orchestrator = OrchestrationFactory.create(
                    technique, retriever=retriever
                )
            else:
                # For AdaptiveRetrieval and others, pass all parameters
                orchestrator = OrchestrationFactory.create(
                    technique,
                    document_id=document_id,
                    top_k=top_k,
                    bm25_weight=bm25_weight,
                    chunking_strategy=chunking_strategy,
                    chunk_size=chunk_size,
                    chunk_overlap=chunk_overlap,
                )

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

            logger.info(f"Orchestration technique {technique} completed successfully")
            return result
        except Exception as e:
            logger.error(f"Orchestration pipeline failed: {e}", exc_info=True)
            raise
