"""
Self-RAG: Self-Reflective Retrieval Augmented Generation

LLM decides if retrieval is needed and evaluates relevance of retrieved documents.
Flow (matching reference implementation):
1. Decision: Should we retrieve?
2. If yes: Retrieve documents
3. Per-document relevance evaluation → filter to relevant contexts
4. Generate response for EACH relevant context (multiple responses)
5. Assess support for each response ("Fully supported", "Partially supported", "No support")
6. Evaluate utility for each response (1-5 rating)
7. Select best response based on support and utility

Performance: Higher latency due to multiple LLM calls (one per relevant document)
"""

import asyncio
import logging
from functools import wraps
from typing import Any, Awaitable, Callable, Dict, List, ParamSpec, TypeVar

from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.retrievers import BaseRetriever

from app.core.dependencies import get_llm
from app.rag.techniques.orchestration.base import BaseOrchestration
from app.rag.techniques.orchestration.models.self_rag_models import (
    RelevanceEvaluation,
    RetrievalDecision,
    SupportAssessment,
    UtilityEvaluation,
)
from app.rag.utils import build_structured_chain_with_fallback

logger = logging.getLogger(__name__)

# Type variables for error handling decorator
P = ParamSpec("P")
T = TypeVar("T")


RETRIEVAL_DECISION_PROMPT = """You are a retrieval decision system. Determine if retrieval is needed to answer the question.
Respond in the same language as the question.

Question: {question}

Determine if retrieval is needed and provide reasoning."""


RELEVANCE_EVALUATION_PROMPT = """Evaluate the relevance of a single retrieved document to the question.
Respond in the same language as the question.

Question: {question}

Document:
{context}

Evaluate whether this document is relevant to the question and provide reasoning."""


GENERATION_PROMPT = """Answer the question based on the provided context. If the context is not relevant or insufficient, say so.
Always respond in the same language as the question (e.g., if question is in Polish, answer in Polish).

Question: {question}

Context:
{context}

Answer:"""


SUPPORT_ASSESSMENT_PROMPT = """Evaluate how well the generated response is supported by the context.
Respond in the same language as the question.

Question: {question}

Context:
{context}

Generated Response:
{response}

Evaluate the support level and provide reasoning."""


UTILITY_EVALUATION_PROMPT = """Rate the utility of the generated response for answering the question.
Respond in the same language as the question.

Question: {question}

Generated Response:
{response}

Rate the utility from 1 (low) to 5 (high) and provide reasoning."""


def with_fallback(default: T):
    """
    Decorator to wrap async methods with error handling and fallback.

    Args:
        default: Default value to return on error

    Returns:
        Decorated function that returns default on error
    """

    def decorator(func: Callable[P, Awaitable[T]]) -> Callable[P, Awaitable[T]]:
        @wraps(func)
        async def wrapper(*args: P.args, **kwargs: P.kwargs) -> T:
            try:
                return await func(*args, **kwargs)
            except Exception as e:
                logger.error(f"Error in {func.__name__}: {e}", exc_info=True)
                return default

        return wrapper

    return decorator


class SelfRAG(BaseOrchestration):
    """
    Self-RAG: LLM decides if retrieval is needed and evaluates relevance.

    Flow (matching reference implementation):
    1. Decision: Should we retrieve?
    2. If yes: Retrieve documents
    3. Per-document relevance evaluation → filter to relevant contexts
    4. Generate response for EACH relevant context (multiple responses)
    5. Assess support for each response ("Fully supported", "Partially supported", "No support")
    6. Evaluate utility for each response (1-5 rating)
    7. Select best response based on support and utility

    Performance: Higher latency due to multiple LLM calls (one per relevant document)
    """

    def __init__(self, retriever: BaseRetriever):
        """
        Initialize Self-RAG.

        Args:
            retriever: Base retriever (Basic or Fusion)
        """
        self.llm = get_llm()
        self.retriever = retriever

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

    async def decide_retrieval(self, question: str) -> RetrievalDecision:
        """
        Step 1: Decide if retrieval is needed.

        Args:
            question: User question

        Returns:
            RetrievalDecision with needs_retrieval and reasoning
        """
        chain = self._build_structured_chain(
            RETRIEVAL_DECISION_PROMPT, RetrievalDecision
        )

        try:
            result = await chain.ainvoke({"question": question})
            return result
        except Exception as e:
            logger.error(f"Error in decide_retrieval: {e}", exc_info=True)
            return RetrievalDecision(
                needs_retrieval=True,
                reasoning=f"Error in decision making: {str(e)}",
            )

    async def evaluate_relevance(
        self, question: str, documents: List[Document]
    ) -> List[Document]:
        """
        Step 2: Evaluate relevance of retrieved documents per-document (parallelized).

        Args:
            question: User question
            documents: Retrieved documents

        Returns:
            List of relevant documents (filtered)
        """
        if not documents:
            return []

        chain = self._build_structured_chain(
            RELEVANCE_EVALUATION_PROMPT, RelevanceEvaluation
        )

        # Create tasks for parallel evaluation
        async def evaluate_document(
            doc: Document, index: int
        ) -> tuple[Document | None, int]:
            """Evaluate a single document and return it if relevant, or None if irrelevant."""
            try:
                result = await chain.ainvoke(
                    {"question": question, "context": doc.page_content}
                )

                if result.relevant:
                    logger.debug(f"Document {index + 1} evaluated as relevant")
                    return doc, index
                else:
                    logger.debug(
                        f"Document {index + 1} evaluated as irrelevant: {result.reasoning}"
                    )
                    return None, index
            except Exception as e:
                logger.error(
                    f"Error evaluating relevance for document {index + 1}: {e}",
                    exc_info=True,
                )
                # On error, include document to be safe
                return doc, index

        # Execute evaluations in parallel
        tasks = [evaluate_document(doc, i) for i, doc in enumerate(documents)]
        results = await asyncio.gather(*tasks, return_exceptions=True)

        # Filter to relevant documents
        relevant_documents = []
        for result in results:
            if isinstance(result, Exception):
                logger.error(f"Exception in parallel evaluation: {result}")
                continue
            doc, _ = result
            if doc is not None:
                relevant_documents.append(doc)

        logger.info(
            f"Relevance evaluation: {len(relevant_documents)}/{len(documents)} documents relevant"
        )
        return relevant_documents

    async def generate_answer(self, question: str, context: str) -> str:
        """
        Step 3: Generate answer from context.

        Args:
            question: User question
            context: Context text

        Returns:
            Generated answer
        """
        chain = self._build_text_chain(GENERATION_PROMPT)
        answer = await chain.ainvoke({"question": question, "context": context})
        return answer

    async def assess_support(
        self, question: str, context: str, response: str
    ) -> SupportAssessment:
        """
        Step 4: Assess support level of the generated response.

        Args:
            question: User question
            context: Context used
            response: Generated response

        Returns:
            SupportAssessment with support_level and reasoning
        """
        chain = self._build_structured_chain(
            SUPPORT_ASSESSMENT_PROMPT, SupportAssessment
        )

        try:
            result = await chain.ainvoke(
                {"question": question, "context": context, "response": response}
            )
            return result
        except Exception as e:
            logger.error(f"Error in assess_support: {e}", exc_info=True)
            return SupportAssessment(
                support_level="Partially supported",
                reasoning=f"Error in support assessment: {str(e)}",
            )

    async def evaluate_utility(self, question: str, response: str) -> UtilityEvaluation:
        """
        Step 5: Evaluate utility of the generated response.

        Args:
            question: User question
            response: Generated response

        Returns:
            UtilityEvaluation with utility_score (1-5) and reasoning
        """
        chain = self._build_structured_chain(
            UTILITY_EVALUATION_PROMPT, UtilityEvaluation
        )

        try:
            result = await chain.ainvoke({"question": question, "response": response})
            return result
        except Exception as e:
            logger.error(f"Error in evaluate_utility: {e}", exc_info=True)
            return UtilityEvaluation(
                utility_score=3,
                reasoning=f"Error in utility evaluation: {str(e)}",
            )

    def _select_best_response(self, responses: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Select best response based on support level and utility score.

        Prioritizes "Fully supported" responses, then highest utility score.

        Args:
            responses: List of response dictionaries with support_level and utility_score

        Returns:
            Best response dictionary
        """
        if not responses:
            raise ValueError("Cannot select from empty responses list")

        # Prioritize "Fully supported", then highest utility score
        best_response = max(
            responses,
            key=lambda x: (
                x["support_level"] == "Fully supported",
                x["utility_score"],
            ),
        )
        return best_response

    async def process(self, question: str, top_k: int = 5) -> Dict[str, Any]:
        """
        Execute full Self-RAG pipeline following reference implementation.

        Flow:
        1. Decide if retrieval is needed
        2. If yes: Retrieve documents
        3. Per-document relevance evaluation → filter to relevant contexts
        4. Generate response for EACH relevant context (multiple responses)
        5. Assess support for each response
        6. Evaluate utility for each response
        7. Select best response based on support and utility

        Args:
            question: User question
            top_k: Number of documents to retrieve

        Returns:
            {
                "answer": str,
                "retrieved": bool,
                "relevance_score": float,
                "critique": str,
                "metadata": dict,
                "documents": list
            }
        """
        metadata = {}

        # Step 1: Decide if retrieval is needed
        try:
            decision = await self.decide_retrieval(question)
            metadata["retrieval_decision"] = {
                "needs_retrieval": decision.needs_retrieval,
                "reasoning": decision.reasoning,
            }
        except Exception as e:
            logger.error(f"Error in retrieval decision: {e}", exc_info=True)
            decision = RetrievalDecision(
                needs_retrieval=True, reasoning=f"Error: {str(e)}"
            )
            metadata["retrieval_decision"] = {
                "needs_retrieval": decision.needs_retrieval,
                "reasoning": decision.reasoning,
            }

        if not decision.needs_retrieval:
            # No retrieval needed - answer from LLM knowledge
            try:
                answer = await self.generate_answer(question, "No retrieval necessary.")
                return {
                    "answer": answer,
                    "retrieved": False,
                    "relevance_score": None,
                    "critique": "No retrieval needed",
                    "metadata": metadata,
                    "documents": [],
                }
            except Exception as e:
                logger.error(
                    f"Error generating answer without retrieval: {e}", exc_info=True
                )
                return {
                    "answer": "I encountered an error while generating a response. Please try again.",
                    "retrieved": False,
                    "relevance_score": None,
                    "critique": f"Error: {str(e)}",
                    "metadata": metadata,
                    "documents": [],
                }

        # Step 2: Retrieve documents
        try:
            documents = await self.retriever.ainvoke(question)
            metadata["documents_retrieved"] = len(documents)
        except Exception as e:
            logger.error(f"Error in document retrieval: {e}", exc_info=True)
            return {
                "answer": "I encountered an error while retrieving documents. Please try again.",
                "retrieved": True,
                "relevance_score": 0.0,
                "critique": f"Retrieval error: {str(e)}",
                "metadata": metadata,
                "documents": [],
            }

        # Check if retrieval returned empty results
        if not documents:
            logger.warning(
                f"Self-RAG: No documents retrieved for query: {question[:50]}"
            )
            try:
                answer = await self.generate_answer(
                    question, "No relevant context found."
                )
            except Exception as e:
                logger.error(
                    f"Error generating answer with no documents: {e}", exc_info=True
                )
                answer = (
                    "I couldn't retrieve any relevant documents from the knowledge base. "
                    "This may be due to: 1) No matching content, 2) Language model limitations, "
                    "or 3) Retrieval configuration. Try rephrasing your query."
                )
            return {
                "answer": answer,
                "retrieved": True,
                "relevance_score": 0.0,
                "critique": "No documents retrieved",
                "metadata": metadata,
                "documents": [],
            }

        # Step 3: Per-document relevance evaluation → filter to relevant contexts
        try:
            relevant_documents = await self.evaluate_relevance(question, documents)
            metadata["relevant_documents_count"] = len(relevant_documents)
            metadata["total_documents_count"] = len(documents)
        except Exception as e:
            logger.error(f"Error in relevance evaluation: {e}", exc_info=True)
            # On error, use all documents
            relevant_documents = documents
            metadata["relevant_documents_count"] = len(relevant_documents)
            metadata["relevance_evaluation_error"] = str(e)

        # If no relevant contexts found, generate without retrieval
        if not relevant_documents:
            logger.info("No relevant contexts found after evaluation")
            try:
                answer = await self.generate_answer(
                    question, "No relevant context found."
                )
            except Exception as e:
                logger.error(
                    f"Error generating answer with no relevant contexts: {e}",
                    exc_info=True,
                )
                answer = "I couldn't find any relevant information in the retrieved documents to answer your question."

            return {
                "answer": answer,
                "retrieved": True,
                "relevance_score": 0.0,
                "critique": "No relevant contexts found",
                "metadata": metadata,
                "documents": documents,  # Include for tracing
            }

        # Step 4: Generate response for EACH relevant context (multiple responses)
        responses = []
        for i, doc in enumerate(relevant_documents):
            try:
                context = f"[Page {doc.metadata.get('page', '?')}] {doc.page_content}"

                # Generate response for this context
                response = await self.generate_answer(question, context)

                # Step 5: Assess support for this response
                support_assessment = await self.assess_support(
                    question, context, response
                )

                # Step 6: Evaluate utility for this response
                utility_evaluation = await self.evaluate_utility(question, response)

                responses.append(
                    {
                        "response": response,
                        "context": context,
                        "document": doc,
                        "support_level": support_assessment.support_level,
                        "support_reasoning": support_assessment.reasoning,
                        "utility_score": utility_evaluation.utility_score,
                        "utility_reasoning": utility_evaluation.reasoning,
                        "index": i,
                    }
                )

                logger.debug(
                    f"Response {i + 1}: support={support_assessment.support_level}, "
                    f"utility={utility_evaluation.utility_score}"
                )

            except Exception as e:
                logger.error(f"Error processing response {i + 1}: {e}", exc_info=True)
                # Continue with other responses
                continue

        # Check if we have any responses
        if not responses:
            logger.warning("No responses generated successfully")
            try:
                answer = await self.generate_answer(
                    question, "Error generating responses."
                )
            except Exception as e:
                logger.error(f"Error in fallback answer generation: {e}", exc_info=True)
                answer = "I encountered an error while generating responses. Please try again."

            return {
                "answer": answer,
                "retrieved": True,
                "relevance_score": 0.0,
                "critique": "Error generating responses",
                "metadata": metadata,
                "documents": documents,
            }

        # Step 7: Select best response based on support and utility
        try:
            best_response = self._select_best_response(responses)

            metadata["response_selection"] = {
                "total_responses": len(responses),
                "selected_index": best_response["index"],
                "selected_support_level": best_response["support_level"],
                "selected_utility_score": best_response["utility_score"],
                "all_responses": [
                    {
                        "index": r["index"],
                        "support_level": r["support_level"],
                        "utility_score": r["utility_score"],
                    }
                    for r in responses
                ],
            }

            # Calculate average relevance score from all relevant documents
            relevance_score = (
                len(relevant_documents) / len(documents) if documents else 0.0
            )

            # Combine support and utility info for critique field
            critique = (
                f"Support: {best_response['support_level']} "
                f"({best_response['support_reasoning']}). "
                f"Utility: {best_response['utility_score']}/5 "
                f"({best_response['utility_reasoning']})"
            )

            return {
                "answer": best_response["response"],
                "retrieved": True,
                "relevance_score": relevance_score,
                "critique": critique,
                "metadata": metadata,
                "documents": documents,  # Include all retrieved documents for tracing
            }

        except Exception as e:
            logger.error(f"Error in response selection: {e}", exc_info=True)
            # Fallback: use first response
            fallback_response = responses[0]
            return {
                "answer": fallback_response["response"],
                "retrieved": True,
                "relevance_score": len(relevant_documents) / len(documents)
                if documents
                else 0.0,
                "critique": f"Selection error, using first response: {str(e)}",
                "metadata": metadata,
                "documents": documents,
            }
