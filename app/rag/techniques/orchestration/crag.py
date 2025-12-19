"""
Corrective RAG (CRAG): Evaluates retrieval quality and uses web search fallback

Flow:
1. Retrieve documents
2. Evaluate relevance score per document
3. Three-way decision:
   - High (>0.7): Use best document
   - Low (<0.3): Web search only
   - Ambiguous (0.3-0.7): Combine best document + web search
4. Generate answer from best source(s)

Requires: Web search tool (DuckDuckGo)
"""

import logging
from typing import Any, Dict, List

from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.retrievers import BaseRetriever

from app.core.dependencies import get_llm
from app.rag.techniques.orchestration.base import BaseOrchestration
from app.rag.techniques.orchestration.models.crag_models import (
    KnowledgeRefinementInput,
    QueryRewriterInput,
    RelevanceScoreInput,
)
from app.services.web_search import web_search
from app.rag.utils import build_structured_chain_with_fallback

logger = logging.getLogger(__name__)


RELEVANCE_SCORER_PROMPT = """On a scale from 0 to 1, how relevant is the following document to the query?
Respond in the same language as the question.

Query: {query}
Document: {document}

Evaluate the relevance and provide reasoning."""

QUERY_REWRITER_PROMPT = """Rewrite the following query to make it more suitable for a web search.
Respond in the same language as the query.

Query: {query}
Rewritten query:"""

KNOWLEDGE_REFINEMENT_PROMPT = """Extract the key information from the following document in bullet points.
Respond in the same language as the document.

Document: {document}
Key points:"""

WEB_SEARCH_PROMPT = """Answer the question using web search results.
Always respond in the same language as the question (e.g., if question is in Polish, answer in Polish).

Question: {question}

Web Search Results:
{web_results}

Answer:"""

COMBINED_ANSWER_PROMPT = """Answer the question based on the combined knowledge from retrieved documents and web search.
Always respond in the same language as the question (e.g., if question is in Polish, answer in Polish).

Question: {question}

Combined Knowledge:
{knowledge}

Sources: {sources}

Answer:"""

DOCUMENT_ANSWER_PROMPT = """Answer the question based on the provided context.
Always respond in the same language as the question (e.g., if question is in Polish, answer in Polish).

Question: {question}

Context:
{context}

Answer:"""


class CRAG(BaseOrchestration):
    """
    Corrective RAG: Evaluates retrieval quality and uses web search fallback.

    Flow:
    1. Retrieve documents
    2. Evaluate relevance score per document
    3. Three-way decision:
       - High (>0.7): Use best document
       - Low (<0.3): Web search only
       - Ambiguous (0.3-0.7): Combine best document + web search
    4. Generate answer from best source(s)

    Requires: Web search tool (DuckDuckGo)
    """

    def __init__(
        self,
        retriever: BaseRetriever,
        relevance_threshold: float = 0.7,
        low_threshold: float = 0.3,
    ):
        """
        Initialize CRAG.

        Args:
            retriever: Base retriever (Basic or Fusion)
            relevance_threshold: High relevance threshold (default 0.7)
            low_threshold: Low relevance threshold (default 0.3)
        """
        self.llm = get_llm()
        self.retriever = retriever
        self.relevance_threshold = relevance_threshold
        self.low_threshold = low_threshold

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

    async def evaluate_documents(
        self, question: str, documents: List[Document]
    ) -> List[float]:
        """
        Evaluate relevance of each document individually.

        Args:
            question: User question
            documents: Retrieved documents

        Returns:
            List of relevance scores (one per document)
        """
        scores = []
        chain = self._build_structured_chain(
            RELEVANCE_SCORER_PROMPT, RelevanceScoreInput
        )

        for doc in documents:
            try:
                result = await chain.ainvoke(
                    {
                        "query": question,
                        "document": doc.page_content[:1000],  # Truncate for prompt
                    }
                )
                scores.append(result.relevance_score)
            except Exception as e:
                logger.warning(
                    f"Error evaluating document relevance: {e}, defaulting to 0.5"
                )
                scores.append(0.5)  # Default to middle score on error

        return scores

    async def rewrite_query(self, query: str) -> str:
        """
        Rewrite query to make it more suitable for web search.

        Args:
            query: Original query

        Returns:
            Rewritten query optimized for web search
        """
        try:
            chain = self._build_structured_chain(
                QUERY_REWRITER_PROMPT, QueryRewriterInput
            )
            result = await chain.ainvoke({"query": query})
            return result.query.strip()
        except Exception as e:
            logger.warning(f"Query rewriting failed: {e}, using original query")
            return query

    async def refine_knowledge(self, sources: List) -> List[str]:
        """
        Extract key points from documents or web search results.

        Args:
            sources: List of Document objects or dicts with 'content' key

        Returns:
            List of refined key points (strings)
        """
        refined_points = []
        chain = self._build_structured_chain(
            KNOWLEDGE_REFINEMENT_PROMPT, KnowledgeRefinementInput
        )

        for source in sources:
            try:
                # Extract content based on source type
                if isinstance(source, Document):
                    content = source.page_content
                elif isinstance(source, dict):
                    content = source.get("content", "") or source.get("title", "")
                else:
                    content = str(source)

                if not content:
                    continue

                result = await chain.ainvoke({"document": content[:2000]})

                # Parse bullet points from the result
                points = [
                    point.strip()
                    for point in result.key_points.split("\n")
                    if point.strip()
                ]
                refined_points.extend(points)
            except Exception as e:
                logger.warning(f"Knowledge refinement failed for source: {e}")
                # Fallback: use original content as single point
                if isinstance(source, Document):
                    refined_points.append(source.page_content[:500])
                elif isinstance(source, dict):
                    refined_points.append(
                        source.get("content", "")[:500] or source.get("title", "")
                    )

        return refined_points

    async def search_web(self, question: str) -> List[Dict]:
        """
        Search web for additional information.

        Args:
            question: User question (can be rewritten)

        Returns:
            List of web search results
        """
        results = await web_search(question, max_results=5)
        return results

    def _format_sources(self, web_results: List[Dict]) -> str:
        """
        Format web search results into a sources string.

        Args:
            web_results: List of web search result dictionaries

        Returns:
            Formatted sources string
        """
        return "\n".join(
            [
                f"{result.get('title', 'Untitled')}: {result.get('url', '')}"
                for result in web_results
                if result.get("url")
            ]
        )

    async def generate_from_web(
        self, question: str, refined_knowledge: List[str], web_results: List[Dict]
    ) -> str:
        """
        Generate answer from refined web search results.

        Args:
            question: User question
            refined_knowledge: Refined key points from web search
            web_results: Original web search results (for source attribution)

        Returns:
            Generated answer
        """
        knowledge_text = "\n".join([f"• {point}" for point in refined_knowledge])
        sources_text = self._format_sources(web_results)

        results_text = (
            f"{knowledge_text}\n\nSources:\n{sources_text}"
            if sources_text
            else knowledge_text
        )

        chain = self._build_text_chain(WEB_SEARCH_PROMPT)
        answer = await chain.ainvoke(
            {"question": question, "web_results": results_text}
        )
        return answer

    async def generate_from_combined(
        self, question: str, combined_knowledge: str, web_results: List[Dict]
    ) -> str:
        """
        Generate answer from combined refined document and web search knowledge.

        Args:
            question: User question
            combined_knowledge: Combined refined knowledge from document and web
            web_results: Web search results (for source attribution)

        Returns:
            Generated answer
        """
        sources_text = self._format_sources(web_results)

        chain = self._build_text_chain(COMBINED_ANSWER_PROMPT)
        answer = await chain.ainvoke(
            {
                "question": question,
                "knowledge": combined_knowledge,
                "sources": sources_text or "Retrieved document",
            }
        )
        return answer

    async def generate_from_documents(
        self, question: str, documents: List[Document]
    ) -> str:
        """
        Generate answer from retrieved documents.

        Args:
            question: User question
            documents: Retrieved documents

        Returns:
            Generated answer
        """
        context = "\n\n".join(
            [
                f"[Page {doc.metadata.get('page', '?')}] {doc.page_content}"
                for doc in documents
            ]
        )

        chain = self._build_text_chain(DOCUMENT_ANSWER_PROMPT)
        answer = await chain.ainvoke({"question": question, "context": context})
        return answer

    async def _handle_empty_retrieval(
        self, question: str, metadata: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Handle case when no documents are retrieved.

        Args:
            question: User question
            metadata: Metadata dictionary to update

        Returns:
            Result dictionary with answer
        """
        logger.warning(f"CRAG: No documents retrieved for query: {question[:50]}")
        metadata["web_search_triggered"] = True
        metadata["web_search_reason"] = "No documents retrieved"

        try:
            rewritten_query = await self.rewrite_query(question)
            web_results = await self.search_web(rewritten_query)
            metadata["web_results_count"] = len(web_results)

            if web_results:
                refined_knowledge = await self.refine_knowledge(web_results)
                answer = await self.generate_from_web(
                    question, refined_knowledge, web_results
                )
                return {
                    "answer": answer,
                    "source": "web",
                    "relevance_score": 0.0,
                    "metadata": metadata,
                    "web_results": web_results,
                }
            else:
                answer = (
                    "I couldn't retrieve relevant information from either the document or web search. "
                    "Please try rephrasing your query or check if the content exists in the document."
                )
                return {
                    "answer": answer,
                    "source": "none",
                    "relevance_score": 0.0,
                    "metadata": metadata,
                }
        except Exception as e:
            logger.error(f"Web search failed: {e}", exc_info=True)
            answer = (
                "I couldn't retrieve any documents and web search failed. "
                "Please try rephrasing your query."
            )
            return {
                "answer": answer,
                "source": "none",
                "relevance_score": 0.0,
                "metadata": metadata,
            }

    async def _handle_evaluation_failure(
        self, question: str, metadata: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Handle case when relevance evaluation fails.

        Args:
            question: User question
            metadata: Metadata dictionary to update

        Returns:
            Result dictionary with answer
        """
        logger.warning("No relevance scores obtained, defaulting to web search")
        metadata["web_search_triggered"] = True
        metadata["web_search_reason"] = "Evaluation failed"

        rewritten_query = await self.rewrite_query(question)
        web_results = await self.search_web(rewritten_query)
        metadata["web_results_count"] = len(web_results)

        if web_results:
            refined_knowledge = await self.refine_knowledge(web_results)
            answer = await self.generate_from_web(
                question, refined_knowledge, web_results
            )
            return {
                "answer": answer,
                "source": "web",
                "relevance_score": 0.0,
                "metadata": metadata,
                "web_results": web_results,
            }
        else:
            answer = "I couldn't retrieve relevant information. Please try rephrasing your query."
            return {
                "answer": answer,
                "source": "none",
                "relevance_score": 0.0,
                "metadata": metadata,
            }

    async def _handle_high_relevance(
        self,
        question: str,
        best_doc: Document,
        max_score: float,
        metadata: Dict[str, Any],
    ) -> Dict[str, Any]:
        """
        Handle high relevance case (>0.7): Use best document.

        Args:
            question: User question
            best_doc: Best document
            max_score: Maximum relevance score
            metadata: Metadata dictionary to update

        Returns:
            Result dictionary with answer
        """
        logger.info(f"High relevance ({max_score:.2f}), using best document")
        metadata["web_search_triggered"] = False
        metadata["decision"] = "high_relevance"
        answer = await self.generate_from_documents(question, [best_doc])

        return {
            "answer": answer,
            "source": "documents",
            "relevance_score": max_score,
            "metadata": metadata,
            "documents": [best_doc],
        }

    async def _handle_low_relevance(
        self,
        question: str,
        max_score: float,
        metadata: Dict[str, Any],
    ) -> Dict[str, Any]:
        """
        Handle low relevance case (<0.3): Web search only.

        Args:
            question: User question
            max_score: Maximum relevance score
            metadata: Metadata dictionary to update

        Returns:
            Result dictionary with answer
        """
        logger.info(f"Low relevance ({max_score:.2f}), using web search only")
        metadata["web_search_triggered"] = True
        metadata["decision"] = "low_relevance"

        rewritten_query = await self.rewrite_query(question)
        metadata["rewritten_query"] = rewritten_query
        web_results = await self.search_web(rewritten_query)
        metadata["web_results_count"] = len(web_results)

        if web_results:
            refined_knowledge = await self.refine_knowledge(web_results)
            answer = await self.generate_from_web(
                question, refined_knowledge, web_results
            )
            return {
                "answer": answer,
                "source": "web",
                "relevance_score": max_score,
                "metadata": metadata,
                "web_results": web_results,
            }
        else:
            answer = "I couldn't retrieve relevant information from web search. Please try rephrasing your query."
            return {
                "answer": answer,
                "source": "none",
                "relevance_score": max_score,
                "metadata": metadata,
            }

    async def _handle_ambiguous_relevance(
        self,
        question: str,
        best_doc: Document,
        max_score: float,
        metadata: Dict[str, Any],
    ) -> Dict[str, Any]:
        """
        Handle ambiguous relevance case (0.3-0.7): Combine best document + web search.

        Args:
            question: User question
            best_doc: Best document
            max_score: Maximum relevance score
            metadata: Metadata dictionary to update

        Returns:
            Result dictionary with answer
        """
        logger.info(
            f"Ambiguous relevance ({max_score:.2f}), combining best document with web search"
        )
        metadata["web_search_triggered"] = True
        metadata["decision"] = "ambiguous_relevance"

        # Rewrite query and perform web search
        rewritten_query = await self.rewrite_query(question)
        metadata["rewritten_query"] = rewritten_query
        web_results = await self.search_web(rewritten_query)
        metadata["web_results_count"] = len(web_results)

        # Refine both retrieved document and web results
        retrieved_knowledge = await self.refine_knowledge([best_doc])
        web_knowledge = await self.refine_knowledge(web_results) if web_results else []

        # Combine knowledge
        combined_knowledge = "\n".join(retrieved_knowledge + web_knowledge)
        metadata["retrieved_knowledge_points"] = len(retrieved_knowledge)
        metadata["web_knowledge_points"] = len(web_knowledge)

        # Generate answer from combined knowledge
        answer = await self.generate_from_combined(
            question, combined_knowledge, web_results
        )

        return {
            "answer": answer,
            "source": "both",
            "relevance_score": max_score,
            "metadata": metadata,
            "documents": [best_doc],
            "web_results": web_results,
        }

    async def process(self, question: str, top_k: int = 5) -> Dict[str, Any]:
        """
        Execute full CRAG pipeline.

        Args:
            question: User question
            top_k: Number of documents to retrieve

        Returns:
            {
                "answer": str,
                "source": "documents" | "web" | "both",
                "relevance_score": float,
                "metadata": dict
            }
        """
        metadata = {}

        # Step 1: Retrieve documents
        documents = await self.retriever.ainvoke(question)
        metadata["documents_retrieved"] = len(documents)

        # Check if retrieval returned empty results
        if not documents:
            return await self._handle_empty_retrieval(question, metadata)

        # Step 2: Evaluate relevance per document
        eval_scores = await self.evaluate_documents(question, documents)
        metadata["relevance_scores"] = eval_scores

        if not eval_scores:
            return await self._handle_evaluation_failure(question, metadata)

        # Step 3: Find max score and best document
        max_score = max(eval_scores)
        best_doc_idx = eval_scores.index(max_score)
        best_doc = documents[best_doc_idx]
        metadata["max_relevance_score"] = max_score
        metadata["best_document_index"] = best_doc_idx

        # Step 4: Three-way decision based on max score
        if max_score > self.relevance_threshold:
            return await self._handle_high_relevance(
                question, best_doc, max_score, metadata
            )
        elif max_score < self.low_threshold:
            return await self._handle_low_relevance(question, max_score, metadata)
        else:
            return await self._handle_ambiguous_relevance(
                question, best_doc, max_score, metadata
            )
