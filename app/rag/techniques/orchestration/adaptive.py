"""
Adaptive Retrieval: Routes queries to different strategies based on query type

Classification (aligned with reference):
- Factual: Query enhancement + LLM ranking
- Analytical: Sub-query generation + LLM diversity selection
- Opinion: Viewpoint identification + LLM opinion selection
- Contextual: Context incorporation + LLM ranking

Each strategy uses LLM to enhance the retrieval process itself.
"""

import logging
from typing import Any, Dict, List, Optional
from uuid import UUID

from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate

from app.core.dependencies import get_llm
from app.rag.techniques.orchestration.base import BaseOrchestration
from app.rag.techniques.orchestration.models.adaptive_models import (
    QueryCategory,
    RelevanceScore,
    SelectedIndices,
    SubQueries,
    Viewpoints,
)
from app.rag.techniques.retrieval.hybrid_retrieval import HybridRetrieval
from app.rag.utils import build_structured_chain_with_fallback

logger = logging.getLogger(__name__)


QUERY_CLASSIFICATION_PROMPT = """Classify the following query into one of these categories: Factual, Analytical, Opinion, or Contextual.
Respond in the same language as the query.

Query: {query}

Category:"""

QUERY_ENHANCEMENT_PROMPT = """Enhance this factual query for better information retrieval.
Respond in the same language as the query.

Query: {query}

Enhanced query:"""

RELEVANCE_RANKING_PROMPT = """On a scale of 1-10, how relevant is this document to the query: '{query}'?
Respond in the same language as the query.

Document: {doc}

Relevance score (1-10):"""

SUB_QUERIES_PROMPT = """Generate {k} sub-questions for comprehensive analysis of the following query.
Respond in the same language as the query.

Query: {query}

Generate a list of {k} distinct sub-questions that will help provide a comprehensive answer to the main query.
Each sub-question should focus on a specific aspect of the topic.

Sub-questions:"""

DIVERSITY_SELECTION_PROMPT = """Select the most diverse and relevant set of {k} documents for the query: '{query}'
Respond in the same language as the query.

Documents:
{docs}

Return only the indices of selected documents as a list of integers (0-based)."""

VIEWPOINTS_PROMPT = """Identify {k} distinct viewpoints or perspectives on the topic: {query}
Respond in the same language as the query.

Viewpoints:"""

OPINION_SELECTION_PROMPT = """Classify these documents into distinct opinions on '{query}' and select the {k} most representative and diverse viewpoints.
Respond in the same language as the query.

Documents:
{docs}

Return only the indices of selected documents as a list of integers (0-based)."""

CONTEXTUAL_QUERY_PROMPT = """Given the user context: {context}
Reformulate the query to best address the user's needs.
Respond in the same language as the query.

Query: {query}

Reformulated query:"""

CONTEXTUAL_RANKING_PROMPT = """Given the query: '{query}' and user context: '{context}', rate the relevance of this document on a scale of 1-10.
Respond in the same language as the query.

Document: {doc}

Relevance score (1-10):"""

GENERATION_PROMPT = """Answer the question based on the provided context.
Always respond in the same language as the question.

Context:
{context}

Question: {question}

Answer:"""


class BaseRetrievalStrategy:
    """Base class for adaptive retrieval strategies."""

    def __init__(
        self,
        document_id: UUID,
        top_k: int = 5,
        bm25_weight: float = 0.5,
        chunking_strategy: str | None = None,
        chunk_size: int | None = None,
        chunk_overlap: int | None = None,
    ):
        """
        Initialize base retrieval strategy.

        Args:
            document_id: UUID of the document to query
            top_k: Number of documents to retrieve
            bm25_weight: BM25 weight for hybrid retrieval
            chunking_strategy: Optional chunking strategy
            chunk_size: Optional chunk size
            chunk_overlap: Optional chunk overlap
        """
        self.document_id = document_id
        self.top_k = top_k
        self.bm25_weight = bm25_weight
        self.chunking_strategy = chunking_strategy
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.llm = get_llm()

    def _build_structured_chain(self, prompt_template: str, model_class: type):
        """
        Build a chain that returns structured output using Pydantic models.

        Uses native structured output if available, falls back to JSON parsing.

        Args:
            prompt_template: Prompt template string
            model_class: Pydantic model class for structured output

        Returns:
            Chain that returns structured output
        """
        return build_structured_chain_with_fallback(
            self.llm, prompt_template, model_class
        )

    def _build_text_chain(self, prompt_template: str):
        """
        Build a chain that returns text output.

        Args:
            prompt_template: Prompt template string

        Returns:
            Chain that returns text
        """
        prompt = ChatPromptTemplate.from_template(prompt_template)
        return prompt | self.llm | StrOutputParser()

    async def retrieve(
        self, query: str, user_context: Optional[str] = None
    ) -> List[Document]:
        """
        Retrieve documents using the strategy.

        Args:
            query: User query
            user_context: Optional user context for contextual queries

        Returns:
            List of retrieved documents
        """
        raise NotImplementedError("Subclasses must implement retrieve method")


class FactualRetrievalStrategy(BaseRetrievalStrategy):
    """Factual strategy: Query enhancement + LLM ranking."""

    async def retrieve(
        self, query: str, user_context: Optional[str] = None
    ) -> List[Document]:
        """
        Retrieve documents for factual queries.

        Flow:
        1. Enhance query using LLM
        2. Retrieve 2×k documents
        3. Rank documents using LLM
        4. Return top k

        Args:
            query: User query
            user_context: Not used for factual queries

        Returns:
            List of top k ranked documents
        """
        logger.info("Retrieving factual query")

        # Step 1: Enhance query using LLM
        enhancement_chain = self._build_text_chain(QUERY_ENHANCEMENT_PROMPT)
        enhanced_query = await enhancement_chain.ainvoke({"query": query})
        logger.info(f"Enhanced query: {enhanced_query}")

        # Step 2: Retrieve documents using enhanced query
        retrieval = HybridRetrieval(
            document_id=self.document_id,
            top_k=self.top_k * 2,
            bm25_weight=self.bm25_weight,
            chunking_strategy=self.chunking_strategy,
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
        )
        docs = await retrieval.retrieve(
            query=enhanced_query,
            document_id=self.document_id,
            top_k=self.top_k * 2,
        )

        if not docs:
            logger.warning("No documents retrieved for factual query")
            return []

        # Step 3: Rank documents using LLM
        ranking_chain = self._build_structured_chain(
            RELEVANCE_RANKING_PROMPT, RelevanceScore
        )

        ranked_docs = []
        logger.info("Ranking documents with LLM")
        for doc in docs:
            try:
                result = await ranking_chain.ainvoke(
                    {"query": enhanced_query, "doc": doc.page_content[:1000]}
                )
                score = float(result.score)
                ranked_docs.append((doc, score))
            except Exception as e:
                logger.warning(f"Error ranking document: {e}")
                # Assign low score on error
                ranked_docs.append((doc, 1.0))

        # Step 4: Sort by relevance score and return top k
        ranked_docs.sort(key=lambda x: x[1], reverse=True)
        return [doc for doc, _ in ranked_docs[: self.top_k]]


class AnalyticalRetrievalStrategy(BaseRetrievalStrategy):
    """Analytical strategy: Sub-query generation + LLM diversity selection."""

    async def retrieve(
        self, query: str, user_context: Optional[str] = None
    ) -> List[Document]:
        """
        Retrieve documents for analytical queries.

        Flow:
        1. Generate k sub-queries using LLM
        2. Retrieve 2 documents per sub-query
        3. Use LLM to select diverse and relevant documents
        4. Return k diverse documents

        Args:
            query: User query
            user_context: Not used for analytical queries

        Returns:
            List of diverse and relevant documents
        """
        logger.info("Retrieving analytical query")

        # Step 1: Generate sub-queries using LLM
        try:
            sub_queries_chain = self._build_structured_chain(
                SUB_QUERIES_PROMPT, SubQueries
            )
            result = await sub_queries_chain.ainvoke({"query": query, "k": self.top_k})
            sub_queries = result.sub_queries
            logger.info(f"Generated sub-queries: {sub_queries}")

            # Validate that we got at least one sub-query
            if not sub_queries or len(sub_queries) == 0:
                logger.warning(
                    "No sub-queries generated, falling back to original query"
                )
                sub_queries = [query]
        except Exception as e:
            logger.error(f"Error generating sub-queries: {e}", exc_info=True)
            logger.warning("Falling back to using original query as single sub-query")
            # Fallback: Use the original query as a single sub-query
            sub_queries = [query]

        # Step 2: Retrieve documents for each sub-query
        retrieval = HybridRetrieval(
            document_id=self.document_id,
            top_k=2,
            bm25_weight=self.bm25_weight,
            chunking_strategy=self.chunking_strategy,
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
        )
        all_docs = []
        for sub_query in sub_queries:
            docs = await retrieval.retrieve(
                query=sub_query,
                document_id=self.document_id,
                top_k=2,
            )
            all_docs.extend(docs)

        if not all_docs:
            logger.warning("No documents retrieved for analytical query")
            return []

        # Step 3: Use LLM to ensure diversity and relevance
        diversity_chain = self._build_structured_chain(
            DIVERSITY_SELECTION_PROMPT, SelectedIndices
        )

        # Format documents for LLM
        docs_text = "\n".join(
            [f"{i}: {doc.page_content[:200]}..." for i, doc in enumerate(all_docs)]
        )

        try:
            result = await diversity_chain.ainvoke(
                {"query": query, "docs": docs_text, "k": self.top_k}
            )
            selected_indices = result.indices
            logger.info(f"Selected diverse documents at indices: {selected_indices}")

            # Filter valid indices
            valid_indices = [i for i in selected_indices if 0 <= i < len(all_docs)]
            if not valid_indices:
                # Fallback: return first k documents
                return all_docs[: self.top_k]

            return [all_docs[i] for i in valid_indices[: self.top_k]]
        except Exception as e:
            logger.error(f"Error in diversity selection: {e}", exc_info=True)
            # Fallback: return first k documents
            return all_docs[: self.top_k]


class OpinionRetrievalStrategy(BaseRetrievalStrategy):
    """Opinion strategy: Viewpoint identification + LLM opinion selection."""

    async def retrieve(
        self, query: str, user_context: Optional[str] = None
    ) -> List[Document]:
        """
        Retrieve documents for opinion queries.

        Flow:
        1. Identify k distinct viewpoints using LLM
        2. Retrieve 2 documents per viewpoint
        3. Use LLM to classify and select diverse opinions
        4. Return k representative viewpoints

        Args:
            query: User query
            user_context: Not used for opinion queries

        Returns:
            List of documents representing diverse opinions
        """
        logger.info("Retrieving opinion query")

        # Step 1: Identify viewpoints using LLM
        viewpoints_chain = self._build_structured_chain(VIEWPOINTS_PROMPT, Viewpoints)
        result = await viewpoints_chain.ainvoke({"query": query, "k": self.top_k})
        viewpoints = result.viewpoints
        logger.info(f"Identified viewpoints: {viewpoints}")

        # Step 2: Retrieve documents for each viewpoint
        retrieval = HybridRetrieval(
            document_id=self.document_id,
            top_k=2,
            bm25_weight=self.bm25_weight,
            chunking_strategy=self.chunking_strategy,
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
        )
        all_docs = []
        for viewpoint in viewpoints:
            # Combine query and viewpoint for retrieval
            combined_query = f"{query} {viewpoint}"
            docs = await retrieval.retrieve(
                query=combined_query,
                document_id=self.document_id,
                top_k=2,
            )
            all_docs.extend(docs)

        if not all_docs:
            logger.warning("No documents retrieved for opinion query")
            return []

        # Step 3: Use LLM to classify and select diverse opinions
        opinion_chain = self._build_structured_chain(
            OPINION_SELECTION_PROMPT, SelectedIndices
        )

        # Format documents for LLM
        docs_text = "\n".join(
            [f"{i}: {doc.page_content[:200]}..." for i, doc in enumerate(all_docs)]
        )

        try:
            result = await opinion_chain.ainvoke(
                {"query": query, "docs": docs_text, "k": self.top_k}
            )
            selected_indices = result.indices
            logger.info(f"Selected opinion documents at indices: {selected_indices}")

            # Filter valid indices
            valid_indices = [i for i in selected_indices if 0 <= i < len(all_docs)]
            if not valid_indices:
                # Fallback: return first k documents
                return all_docs[: self.top_k]

            return [all_docs[i] for i in valid_indices[: self.top_k]]
        except Exception as e:
            logger.error(f"Error in opinion selection: {e}", exc_info=True)
            # Fallback: return first k documents
            return all_docs[: self.top_k]


class ContextualRetrievalStrategy(BaseRetrievalStrategy):
    """Contextual strategy: Context incorporation + LLM ranking."""

    async def retrieve(
        self, query: str, user_context: Optional[str] = None
    ) -> List[Document]:
        """
        Retrieve documents for contextual queries.

        Flow:
        1. Incorporate user context into query using LLM
        2. Retrieve 2×k documents
        3. Rank documents considering both relevance and context
        4. Return top k

        Args:
            query: User query
            user_context: Optional user context

        Returns:
            List of top k ranked documents
        """
        logger.info("Retrieving contextual query")

        # Step 1: Incorporate user context into query
        context_chain = self._build_text_chain(CONTEXTUAL_QUERY_PROMPT)
        contextualized_query = await context_chain.ainvoke(
            {
                "query": query,
                "context": user_context or "No specific context provided",
            }
        )
        logger.info(f"Contextualized query: {contextualized_query}")

        # Step 2: Retrieve documents using contextualized query
        retrieval = HybridRetrieval(
            document_id=self.document_id,
            top_k=self.top_k * 2,
            bm25_weight=self.bm25_weight,
            chunking_strategy=self.chunking_strategy,
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
        )
        docs = await retrieval.retrieve(
            query=contextualized_query,
            document_id=self.document_id,
            top_k=self.top_k * 2,
        )

        if not docs:
            logger.warning("No documents retrieved for contextual query")
            return []

        # Step 3: Rank documents considering both relevance and context
        ranking_chain = self._build_structured_chain(
            CONTEXTUAL_RANKING_PROMPT, RelevanceScore
        )

        ranked_docs = []
        logger.info("Ranking documents with LLM (considering context)")
        for doc in docs:
            try:
                result = await ranking_chain.ainvoke(
                    {
                        "query": contextualized_query,
                        "context": user_context or "No specific context provided",
                        "doc": doc.page_content[:1000],
                    }
                )
                score = float(result.score)
                ranked_docs.append((doc, score))
            except Exception as e:
                logger.warning(f"Error ranking document: {e}")
                # Assign low score on error
                ranked_docs.append((doc, 1.0))

        # Step 4: Sort by relevance score and return top k
        ranked_docs.sort(key=lambda x: x[1], reverse=True)
        return [doc for doc, _ in ranked_docs[: self.top_k]]


class AdaptiveRetrieval(BaseOrchestration):
    """
    Adaptive Retrieval: Routes queries to different strategies based on query type.

    Classification (aligned with reference):
    - Factual: Query enhancement + LLM ranking
    - Analytical: Sub-query generation + LLM diversity selection
    - Opinion: Viewpoint identification + LLM opinion selection
    - Contextual: Context incorporation + LLM ranking

    Each strategy uses LLM to enhance the retrieval process itself.
    """

    def __init__(
        self,
        document_id: UUID,
        top_k: int = 5,
        bm25_weight: float = 0.5,
        chunking_strategy: str | None = None,
        chunk_size: int | None = None,
        chunk_overlap: int | None = None,
    ):
        """
        Initialize Adaptive Retrieval.

        Args:
            document_id: UUID of the document to query
            top_k: Number of documents to retrieve
            bm25_weight: BM25 weight for hybrid retrieval
            chunking_strategy: Optional chunking strategy
            chunk_size: Optional chunk size
            chunk_overlap: Optional chunk overlap
        """
        self.llm = get_llm()
        self.document_id = document_id
        self.top_k = top_k
        self.bm25_weight = bm25_weight
        self.chunking_strategy = chunking_strategy
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

        # Initialize strategies
        self.strategies = {
            "Factual": FactualRetrievalStrategy(
                document_id,
                top_k,
                bm25_weight,
                chunking_strategy,
                chunk_size,
                chunk_overlap,
            ),
            "Analytical": AnalyticalRetrievalStrategy(
                document_id,
                top_k,
                bm25_weight,
                chunking_strategy,
                chunk_size,
                chunk_overlap,
            ),
            "Opinion": OpinionRetrievalStrategy(
                document_id,
                top_k,
                bm25_weight,
                chunking_strategy,
                chunk_size,
                chunk_overlap,
            ),
            "Contextual": ContextualRetrievalStrategy(
                document_id,
                top_k,
                bm25_weight,
                chunking_strategy,
                chunk_size,
                chunk_overlap,
            ),
        }

    def _build_structured_chain(self, prompt_template: str, model_class: type):
        """
        Build a chain that returns structured output using Pydantic models.

        Uses native structured output if available, falls back to JSON parsing.

        Args:
            prompt_template: Prompt template string
            model_class: Pydantic model class for structured output

        Returns:
            Chain that returns structured output
        """
        return build_structured_chain_with_fallback(
            self.llm, prompt_template, model_class
        )

    async def classify_query(self, query: str) -> str:
        """
        Classify query to determine best strategy.

        Args:
            query: User query

        Returns:
            Category: "Factual", "Analytical", "Opinion", or "Contextual"
        """
        logger.info("Classifying query")
        classification_chain = self._build_structured_chain(
            QUERY_CLASSIFICATION_PROMPT, QueryCategory
        )

        try:
            result = await classification_chain.ainvoke({"query": query})
            category = result.category
            logger.info(f"Query classified as: {category}")
            return category
        except Exception as e:
            logger.error(f"Error in classify_query: {e}", exc_info=True)
            # Default to Factual on error
            return "Factual"

    async def process(
        self,
        query: str,
        top_k: Optional[int] = None,
        bm25_weight: Optional[float] = None,
        temperature: float = 0.7,
        user_context: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Execute adaptive retrieval pipeline.

        Args:
            query: User query
            top_k: Number of documents to retrieve (overrides init if provided)
            bm25_weight: BM25 weight for hybrid retrieval (overrides init if provided)
            temperature: LLM temperature
            user_context: Optional user context for contextual queries

        Returns:
            {
                "answer": str,
                "strategy_used": str,
                "classification": str,
                "metadata": dict,
                "documents": list
            }
        """
        # Use provided parameters or fall back to instance defaults
        effective_top_k = top_k if top_k is not None else self.top_k
        effective_bm25_weight = (
            bm25_weight if bm25_weight is not None else self.bm25_weight
        )

        # Update strategies if parameters changed
        if top_k is not None or bm25_weight is not None:
            self.strategies = {
                "Factual": FactualRetrievalStrategy(
                    self.document_id, effective_top_k, effective_bm25_weight
                ),
                "Analytical": AnalyticalRetrievalStrategy(
                    self.document_id, effective_top_k, effective_bm25_weight
                ),
                "Opinion": OpinionRetrievalStrategy(
                    self.document_id, effective_top_k, effective_bm25_weight
                ),
                "Contextual": ContextualRetrievalStrategy(
                    self.document_id, effective_top_k, effective_bm25_weight
                ),
            }

        # Step 1: Classify query
        category = await self.classify_query(query)

        # Step 2: Get appropriate strategy
        strategy = self.strategies.get(category, self.strategies["Factual"])

        # Step 3: Retrieve documents using strategy
        documents = await strategy.retrieve(query, user_context)

        if not documents:
            logger.warning("No documents retrieved, returning empty answer")
            return {
                "answer": "I couldn't find relevant information to answer your question.",
                "strategy_used": category.lower(),
                "classification": category,
                "metadata": {
                    "documents_retrieved": 0,
                },
                "documents": [],
            }

        # Step 4: Generate answer
        context = "\n\n".join(
            [
                f"[Page {doc.metadata.get('page', '?')}] {doc.page_content}"
                for doc in documents
            ]
        )

        llm = get_llm()
        llm.temperature = temperature

        prompt = ChatPromptTemplate.from_template(GENERATION_PROMPT)
        chain = prompt | llm | StrOutputParser()

        answer = await chain.ainvoke({"question": query, "context": context})

        return {
            "answer": answer,
            "strategy_used": category.lower(),
            "classification": category,
            "metadata": {
                "documents_retrieved": len(documents),
            },
            "documents": documents,  # Include for tracing
        }
