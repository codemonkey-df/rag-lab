"""
HyDE (Hypothetical Document Embedding) query expansion technique
"""

import logging
import time

from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate

from app.core.dependencies import get_hyde_llm
from app.rag.techniques.query_expansion.base import BaseQueryExpansion

logger = logging.getLogger(__name__)


HYDE_PROMPT = """Generate a concise hypothetical document (1-2 paragraphs, max 150 words) that would answer this question.
Write as if from a relevant source. Use the SAME LANGUAGE as the question.

Question: {question}

Hypothetical Document:"""


class HyDEExpansion(BaseQueryExpansion):
    """
    HyDE (Hypothetical Document Embedding) query expansion.

    Generates a hypothetical document that would answer the query,
    then uses that document for better embedding alignment.
    Uses a smaller, faster model for query expansion.
    """

    async def expand(self, query: str) -> str:
        """
        Expand query using HyDE.

        Args:
            query: Original user query

        Returns:
            Hypothetical document text
        """
        start_time = time.time()
        llm = get_hyde_llm()  # Use smaller model for faster HyDE
        logger.info(f"Starting HyDE expansion with model: {llm.model}")

        prompt = ChatPromptTemplate.from_template(HYDE_PROMPT)
        chain = prompt | llm | StrOutputParser()
        hypothetical_doc = await chain.ainvoke({"question": query})

        elapsed = (time.time() - start_time) * 1000
        logger.info(
            f"HyDE expansion completed in {elapsed:.0f}ms, generated {len(hypothetical_doc)} chars"
        )

        # Warn if HyDE takes too long (suggests Ollama concurrency issue)
        if elapsed > 10000:  # 10 seconds
            logger.warning(
                f"HyDE took {elapsed:.0f}ms (>10s). This suggests Ollama may not be configured for parallel processing. "
                f"Ensure OLLAMA_NUM_PARALLEL=4 and OLLAMA_MAX_LOADED_MODELS=2 are set BEFORE starting Ollama, "
                f"then restart Ollama server."
            )

        return hypothetical_doc


async def expand_query_with_hyde(query: str) -> str:
    """
    Expand query using HyDE (Hypothetical Document Embedding).

    DEPRECATED: Use HyDEExpansion class instead.
    This function is kept for backward compatibility.

    Generates a hypothetical document that would answer the query,
    then uses that document for better embedding alignment.
    Uses a smaller, faster model for query expansion.

    Args:
        query: Original user query

    Returns:
        Hypothetical document text
    """
    expansion = HyDEExpansion()
    return await expansion.expand(query)
