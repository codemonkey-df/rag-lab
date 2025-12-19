"""
Proposition chunking indexing strategy with atomic fact extraction.

Breaks documents into atomic, factual propositions using LLM-based extraction
and quality checking. Each proposition is a self-contained, factual statement.

Research: Tony Chen, et. al. (https://doi.org/10.48550/arXiv.2312.06648)
"""

import logging
from typing import Any, Dict, List, Optional
from uuid import UUID

from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate, FewShotChatMessagePromptTemplate
from langchain_text_splitters import RecursiveCharacterTextSplitter
from pydantic import BaseModel, Field

from app.core.dependencies import get_llm
from app.rag.techniques.indexing.base import BaseIndexingStrategy
from app.rag.techniques.indexing.standard import add_line_numbers_to_chunks
from app.services.vectorstore import get_chroma_collection

logger = logging.getLogger(__name__)


# ============================================================================
# PROPOSITION HELPER CLASSES AND FUNCTIONS
# ============================================================================


class GeneratePropositions(BaseModel):
    """List of all the propositions in a given document chunk"""

    propositions: List[str] = Field(
        description="List of propositions (factual, self-contained, and concise information)"
    )


class GradePropositions(BaseModel):
    """Grade a given proposition on accuracy, clarity, completeness, and conciseness"""

    accuracy: int = Field(
        description="Rate from 1-10 based on how well the proposition reflects the original text.",
        ge=1,
        le=10,
    )

    clarity: int = Field(
        description="Rate from 1-10 based on how easy it is to understand the proposition without additional context.",
        ge=1,
        le=10,
    )

    completeness: int = Field(
        description="Rate from 1-10 based on whether the proposition includes necessary details (e.g., dates, qualifiers).",
        ge=1,
        le=10,
    )

    conciseness: int = Field(
        description="Rate from 1-10 based on whether the proposition is concise without losing important information.",
        ge=1,
        le=10,
    )


# Prompts
PROPOSITION_SYSTEM_PROMPT = """Please break down the following text into simple, self-contained propositions. Ensure that each proposition meets the following criteria:

1. Express a Single Fact: Each proposition should state one specific fact or claim.
2. Be Understandable Without Context: The proposition should be self-contained, meaning it can be understood without needing additional context.
3. Use Full Names, Not Pronouns: Avoid pronouns or ambiguous references; use full entity names.
4. Include Relevant Dates/Qualifiers: If applicable, include necessary dates, times, and qualifiers to make the fact precise.
5. Contain One Subject-Predicate Relationship: Focus on a single subject and its corresponding action or attribute, without conjunctions or multiple clauses."""

PROPOSITION_EXAMPLES = [
    {
        "document": "In 1969, Neil Armstrong became the first person to walk on the Moon during the Apollo 11 mission.",
        "propositions": "['Neil Armstrong was an astronaut.', 'Neil Armstrong walked on the Moon in 1969.', 'Neil Armstrong was the first person to walk on the Moon.', 'Neil Armstrong walked on the Moon during the Apollo 11 mission.', 'The Apollo 11 mission occurred in 1969.']",
    },
]

EVALUATION_PROMPT_TEMPLATE = """
Please evaluate the following proposition based on the criteria below:
- **Accuracy**: Rate from 1-10 based on how well the proposition reflects the original text.
- **Clarity**: Rate from 1-10 based on how easy it is to understand the proposition without additional context.
- **Completeness**: Rate from 1-10 based on whether the proposition includes necessary details (e.g., dates, qualifiers).
- **Conciseness**: Rate from 1-10 based on whether the proposition is concise without losing important information.

Example:
Docs: In 1969, Neil Armstrong became the first person to walk on the Moon during the Apollo 11 mission.

Proposition_1: Neil Armstrong was an astronaut.
Evaluation_1: "accuracy": 10, "clarity": 10, "completeness": 10, "conciseness": 10

Proposition_2: Neil Armstrong walked on the Moon in 1969.
Evaluation_2: "accuracy": 10, "clarity": 10, "completeness": 10, "conciseness": 10

Proposition_3: Neil Armstrong was the first person to walk on the Moon.
Evaluation_3: "accuracy": 10, "clarity": 10, "completeness": 10, "conciseness": 10

Proposition_4: Neil Armstrong walked on the Moon during the Apollo 11 mission.
Evaluation_4: "accuracy": 10, "clarity": 10, "completeness": 10, "conciseness": 10

Proposition_5: The Apollo 11 mission occurred in 1969.
Evaluation_5: "accuracy": 10, "clarity": 10, "completeness": 10, "conciseness": 10

Format:
Proposition: "{proposition}"
Original Text: "{original_text}"
"""


def _build_proposition_generator():
    """
    Build a chain that generates propositions from text using structured output.

    Returns:
        Chain that takes document text and returns GeneratePropositions
    """
    llm = get_llm()
    structured_llm = llm.with_structured_output(GeneratePropositions)

    # Few-shot examples
    example_prompt = ChatPromptTemplate.from_messages(
        [
            ("human", "{document}"),
            ("ai", "{propositions}"),
        ]
    )

    few_shot_prompt = FewShotChatMessagePromptTemplate(
        example_prompt=example_prompt,
        examples=PROPOSITION_EXAMPLES,
    )

    # Main prompt
    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", PROPOSITION_SYSTEM_PROMPT),
            few_shot_prompt,
            ("human", "{document}"),
        ]
    )

    return prompt | structured_llm


def _build_proposition_evaluator():
    """
    Build a chain that evaluates proposition quality using structured output.

    Returns:
        Chain that takes proposition and original text, returns GradePropositions
    """
    llm = get_llm()
    structured_llm = llm.with_structured_output(GradePropositions)

    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", EVALUATION_PROMPT_TEMPLATE),
            ("human", "Proposition: {proposition}\nOriginal Text: {original_text}"),
        ]
    )

    return prompt | structured_llm


async def generate_propositions(chunk_text: str) -> List[str]:
    """
    Generate propositions from a chunk using LLM with structured output.

    Args:
        chunk_text: Text content of the chunk

    Returns:
        List of generated propositions (factual, self-contained statements)
    """
    try:
        generator = _build_proposition_generator()
        result = await generator.ainvoke({"document": chunk_text})
        return result.propositions
    except Exception as e:
        logger.warning(f"Failed to generate propositions for chunk: {e}")
        return []


async def evaluate_proposition(proposition: str, original_text: str) -> Dict[str, int]:
    """
    Evaluate proposition quality on four metrics.

    Args:
        proposition: The proposition to evaluate
        original_text: The original chunk text for comparison

    Returns:
        Dictionary with scores: {"accuracy": int, "clarity": int, "completeness": int, "conciseness": int}
    """
    try:
        evaluator = _build_proposition_evaluator()
        result = await evaluator.ainvoke(
            {"proposition": proposition, "original_text": original_text}
        )
        return {
            "accuracy": result.accuracy,
            "clarity": result.clarity,
            "completeness": result.completeness,
            "conciseness": result.conciseness,
        }
    except Exception as e:
        logger.warning(f"Failed to evaluate proposition '{proposition[:50]}...': {e}")
        # Return low scores on failure
        return {"accuracy": 0, "clarity": 0, "completeness": 0, "conciseness": 0}


def passes_quality_check(scores: Dict[str, int], thresholds: Dict[str, int]) -> bool:
    """
    Check if proposition passes all quality thresholds.

    Args:
        scores: Quality scores from evaluate_proposition
        thresholds: Minimum scores required for each metric

    Returns:
        True if all scores meet or exceed thresholds
    """
    for category, threshold in thresholds.items():
        if scores.get(category, 0) < threshold:
            return False
    return True


async def filter_propositions(
    propositions: List[Document],
    original_chunks: List[Document],
    thresholds: Optional[Dict[str, int]] = None,
) -> List[Document]:
    """
    Filter propositions based on quality thresholds.

    Args:
        propositions: List of proposition Documents to filter
        original_chunks: List of original chunk Documents (for evaluation)
        thresholds: Quality thresholds (default: all 7)

    Returns:
        List of Documents that pass quality checks
    """
    if thresholds is None:
        thresholds = {"accuracy": 7, "clarity": 7, "completeness": 7, "conciseness": 7}

    # Build chunk lookup by chunk_id
    chunk_lookup = {}
    for chunk in original_chunks:
        chunk_id = chunk.metadata.get("chunk_id")
        if chunk_id:
            chunk_lookup[chunk_id] = chunk

    filtered_propositions = []

    for proposition_doc in propositions:
        chunk_id = proposition_doc.metadata.get("chunk_id")
        if not chunk_id:
            logger.warning(
                f"Proposition missing chunk_id, skipping: {proposition_doc.page_content[:50]}"
            )
            continue

        original_chunk = chunk_lookup.get(chunk_id)
        if not original_chunk:
            logger.warning(
                f"Original chunk {chunk_id} not found for proposition, skipping"
            )
            continue

        # Evaluate proposition quality
        scores = await evaluate_proposition(
            proposition_doc.page_content, original_chunk.page_content
        )

        # Check if passes thresholds
        if passes_quality_check(scores, thresholds):
            filtered_propositions.append(proposition_doc)
        else:
            logger.debug(
                f"Proposition failed quality check (scores: {scores}): {proposition_doc.page_content[:50]}"
            )

    return filtered_propositions


# ============================================================================
# PROPOSITION STRATEGY CLASS
# ============================================================================


class PropositionStrategy(BaseIndexingStrategy):
    """
    Proposition indexing strategy with atomic fact extraction.

    Process:
    1. Initial chunking with small chunks (200 chars, 50 overlap)
    2. Generate propositions from each chunk using LLM
    3. Quality check each proposition (accuracy, clarity, completeness, conciseness)
    4. Filter propositions that meet quality thresholds
    5. Store propositions in ChromaDB

    Characteristics:
    - Atomic facts: Each proposition is a single, factual statement
    - High precision: Quality-checked using LLM evaluation
    - Expensive: 2-3 LLM calls per chunk (generation + quality check)
    - Very slow: 20-page PDF ≈ 60+ minutes

    Performance:
    - 100+ page PDF: 60+ minutes (1-2 LLM calls per chunk)
    - Runs as background task (async execution)
    - WARNING: Extremely slow on local LLMs

    Configuration:
    - chunk_size: Ignored, uses hardcoded 200 for initial chunking
    - chunk_overlap: Ignored, uses hardcoded 50 for initial chunking
    - quality_thresholds: Dict[str, int] with accuracy, clarity, completeness, conciseness
      (default: all 7 out of 10)
    """

    # Hardcoded initial chunk sizes for proposition generation
    INITIAL_CHUNK_SIZE = 200
    INITIAL_CHUNK_OVERLAP = 50

    async def chunk(
        self,
        documents: List[Document],
        config: Dict[str, Any],
    ) -> List[Document]:
        """
        Initial chunking for proposition generation.

        Uses smaller chunks (200 chars) to generate more granular propositions.
        The actual propositions become the final chunks.

        Args:
            documents: Raw documents from PDF loader
            config: Ignored for initial chunking

        Returns:
            List of initially chunked documents
        """
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.INITIAL_CHUNK_SIZE,
            chunk_overlap=self.INITIAL_CHUNK_OVERLAP,
            length_function=len,
        )
        chunks = splitter.split_documents(documents)

        # Add chunk_id metadata for tracking
        for i, chunk in enumerate(chunks):
            chunk.metadata["chunk_id"] = i + 1

        logger.info(
            f"Proposition initial chunking: {len(documents)} docs -> "
            f"{len(chunks)} initial chunks"
        )

        return chunks

    async def post_process(
        self,
        chunks: List[Document],
        config: Dict[str, Any],
    ) -> List[Document]:
        """
        Post-process: generate and filter propositions.

        WARNING: This is extremely slow - 1-2 LLM calls per chunk.

        Process:
        1. Generate propositions from each chunk
        2. Evaluate proposition quality
        3. Filter based on quality thresholds
        4. Return propositions that pass quality checks

        Args:
            chunks: Initially chunked documents
            config: Optional quality_thresholds and progress_callback

        Returns:
            List of quality-checked proposition documents
        """
        quality_thresholds = config.get(
            "quality_thresholds",
            {"accuracy": 7, "clarity": 7, "completeness": 7, "conciseness": 7},
        )

        progress_callback = config.get("progress_callback")

        # Generate propositions from each chunk
        all_propositions = []
        total_chunks = len(chunks)

        for i, chunk in enumerate(chunks):
            try:
                # Generate propositions from chunk
                propositions_text = await generate_propositions(chunk.page_content)

                # Create Document objects for each proposition
                for prop_text in propositions_text:
                    prop_metadata = chunk.metadata.copy()
                    prop_doc = Document(page_content=prop_text, metadata=prop_metadata)
                    all_propositions.append(prop_doc)

                # Update progress
                if (i + 1) % max(1, total_chunks // 10) == 0 or (i + 1) == total_chunks:
                    progress_pct = ((i + 1) / total_chunks) * 100
                    if progress_callback:
                        progress_callback(progress_pct)

                logger.debug(
                    f"Generated {len(propositions_text)} propositions from chunk {i + 1}"
                )

            except Exception as e:
                logger.warning(
                    f"Failed to generate propositions for chunk {i + 1}: {e}"
                )
                continue

        logger.info(f"Generated {len(all_propositions)} total propositions")

        # Quality check and filter propositions
        filtered_propositions = await filter_propositions(
            all_propositions, chunks, quality_thresholds
        )

        logger.info(
            f"After quality filtering: {len(filtered_propositions)} propositions "
            f"passed (from {len(all_propositions)} generated)"
        )

        # If no propositions passed quality check, fall back to all propositions
        if not filtered_propositions and all_propositions:
            logger.warning(
                "No propositions passed quality check, falling back to all generated propositions"
            )
            filtered_propositions = all_propositions

        # Add line numbers to propositions
        filtered_propositions = add_line_numbers_to_chunks(filtered_propositions)

        return filtered_propositions

    async def index(
        self,
        document_id: UUID,
        chunks: List[Document],
    ) -> None:
        """
        Index proposition documents into ChromaDB.

        Args:
            document_id: Document UUID for collection name
            chunks: Proposition documents to index

        Raises:
            Exception: If ChromaDB operation fails
        """
        collection = get_chroma_collection(document_id)
        collection.add_documents(chunks)
        logger.info(
            f"Indexed {len(chunks)} propositions to ChromaDB for doc {document_id}"
        )

    def supports_async_execution(self) -> bool:
        """Proposition strategy is extremely slow, must run as background task."""
        return True

    def get_optional_config(self) -> Dict[str, Any]:
        """Provide defaults for proposition strategy."""
        return {
            "quality_thresholds": {
                "accuracy": 7,
                "clarity": 7,
                "completeness": 7,
                "conciseness": 7,
            },
            "progress_callback": None,
        }
