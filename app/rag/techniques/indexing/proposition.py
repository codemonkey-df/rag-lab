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
from langchain_text_splitters import RecursiveCharacterTextSplitter
from pydantic import BaseModel, Field

from app.core.config import get_settings
from app.core.dependencies import get_llm
from app.rag.techniques.indexing.base import BaseIndexingStrategy
from app.rag.techniques.indexing.standard import add_line_numbers_to_chunks
from app.rag.techniques.indexing.utils import process_items_in_parallel
from app.rag.utils import build_structured_chain_with_fallback
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

    # Build prompt as string template with few-shot examples
    # Format few-shot examples
    few_shot_examples = "\n\n".join(
        [
            f"Document: {ex['document']}\nPropositions: {ex['propositions']}"
            for ex in PROPOSITION_EXAMPLES
        ]
    )

    # Build complete prompt template as string
    prompt_template = f"""{PROPOSITION_SYSTEM_PROMPT}

Examples:
{few_shot_examples}

Document: {{document}}

Generate propositions for the above document."""

    # Use fallback utility that handles Ollama LLMs
    return build_structured_chain_with_fallback(
        llm=llm,
        prompt_template=prompt_template,
        model_class=GeneratePropositions,
    )


def _build_proposition_evaluator():
    """
    Build a chain that evaluates proposition quality using structured output.

    Returns:
        Chain that takes proposition and original text, returns GradePropositions
    """
    llm = get_llm()

    # Build prompt template - match reference format (comma-separated)
    prompt_template = f"""{EVALUATION_PROMPT_TEMPLATE}

{{proposition}}, {{original_text}}"""

    # Use fallback utility that handles Ollama LLMs
    return build_structured_chain_with_fallback(
        llm=llm,
        prompt_template=prompt_template,
        model_class=GradePropositions,
    )


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
        propositions = result.propositions if result.propositions else []

        if not propositions:
            logger.debug(
                f"LLM returned empty proposition list for chunk "
                f"(length: {len(chunk_text)} chars)"
            )
        else:
            logger.debug(
                f"Generated {len(propositions)} propositions from chunk "
                f"(length: {len(chunk_text)} chars)"
            )

        return propositions
    except Exception as e:
        logger.warning(
            f"Failed to generate propositions for chunk (length: {len(chunk_text)} chars): {e}",
            exc_info=True,
        )
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

    if not propositions:
        logger.warning("No propositions provided for quality filtering")
        return []

    # Build chunk lookup by chunk_id
    chunk_lookup = {}
    for chunk in original_chunks:
        chunk_id = chunk.metadata.get("chunk_id")
        if chunk_id:
            chunk_lookup[chunk_id] = chunk

    # Filter out invalid propositions before parallel processing
    valid_propositions = []
    skipped_count = 0

    for proposition_doc in propositions:
        # Skip empty propositions (defensive check)
        if not proposition_doc.page_content or not proposition_doc.page_content.strip():
            skipped_count += 1
            logger.debug("Skipping empty proposition during quality filtering")
            continue

        chunk_id = proposition_doc.metadata.get("chunk_id")
        if not chunk_id:
            skipped_count += 1
            logger.warning(
                f"Proposition missing chunk_id, skipping: {proposition_doc.page_content[:50]}"
            )
            continue

        original_chunk = chunk_lookup.get(chunk_id)
        if not original_chunk:
            skipped_count += 1
            logger.warning(
                f"Original chunk {chunk_id} not found for proposition, skipping"
            )
            continue

        valid_propositions.append((proposition_doc, original_chunk))

    if skipped_count > 0:
        logger.debug(
            f"Skipped {skipped_count} propositions during quality filtering "
            f"(missing metadata or empty content)"
        )

    if not valid_propositions:
        logger.warning("No valid propositions to evaluate after filtering")
        return []

    # Evaluate propositions in parallel
    settings = get_settings()

    async def evaluate_one(item: tuple, index: int):
        """Evaluate a single proposition."""
        proposition_doc, original_chunk = item
        scores = await evaluate_proposition(
            proposition_doc.page_content, original_chunk.page_content
        )
        return scores, proposition_doc

    results = await process_items_in_parallel(
        items=valid_propositions,
        process_func=evaluate_one,
        max_concurrent=settings.max_concurrent_evaluations,
        item_name="proposition",
    )

    # Process results and filter by quality thresholds
    filtered_propositions = []
    for result_tuple, index, error in results:
        if error is not None:
            logger.warning(f"Failed to evaluate proposition {index}: {error}")
            continue

        # Unpack the result tuple
        scores, proposition_doc = result_tuple

        # Check if passes thresholds
        if passes_quality_check(scores, thresholds):
            filtered_propositions.append(proposition_doc)
        else:
            logger.debug(
                f"Proposition failed quality check (scores: {scores}, "
                f"thresholds: {thresholds}): {proposition_doc.page_content[:50]}"
            )

    return filtered_propositions


# ============================================================================
# PROPOSITION STRATEGY CLASS
# ============================================================================


class PropositionStrategy(BaseIndexingStrategy):
    """
    Proposition indexing strategy with atomic fact extraction.

    Process:
    1. Initial chunking with configurable chunk size and overlap
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
    - chunk_size: Chunk size for initial chunking (default: 1024)
    - chunk_overlap: Chunk overlap for initial chunking (default: 200)
    - quality_thresholds: Dict[str, int] with accuracy, clarity, completeness, conciseness
      (default: all 7 out of 10)
    """

    async def chunk(
        self,
        documents: List[Document],
        config: Dict[str, Any],
    ) -> List[Document]:
        """
        Initial chunking for proposition generation.

        Uses configurable chunk size and overlap to generate granular propositions.
        The actual propositions become the final chunks.

        Args:
            documents: Raw documents from PDF loader
            config: Must contain 'chunk_size' and 'chunk_overlap'

        Returns:
            List of initially chunked documents
        """
        self.validate_config(config)

        chunk_size = config["chunk_size"]
        chunk_overlap = config["chunk_overlap"]
        if chunk_overlap >= 200:
            logger.warning(
                "Chunk overlap is greater than 200, which may result in poor proposition generation. "
                "Consider using a smaller chunk overlap."
            )

        splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
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
        settings = get_settings()
        
        quality_thresholds = {
            "accuracy": settings.proposition_quality_threshold_accuracy,
            "clarity": settings.proposition_quality_threshold_clarity,
            "completeness": settings.proposition_quality_threshold_completeness,
            "conciseness": settings.proposition_quality_threshold_conciseness,
        }

        progress_callback = config.get("progress_callback")

        # Generate propositions from each chunk in parallel
        all_propositions = []
        total_chunks = len(chunks)
        

        async def process_chunk(chunk: Document, index: int):
            """Process a single chunk to generate propositions."""
            propositions_text = await generate_propositions(chunk.page_content)
            return propositions_text, chunk

        # Process chunks in parallel
        results = await process_items_in_parallel(
            items=chunks,
            process_func=process_chunk,
            max_concurrent=settings.max_concurrent_chunks,
            item_name="chunk",
        )

        # Process results and create proposition documents
        for result_tuple, index, error in results:
            if error is not None:
                logger.warning(
                    f"Failed to generate propositions for chunk {index + 1}: {error}"
                )
                continue

            # Unpack the result tuple
            propositions_text, chunk = result_tuple

            if not propositions_text:
                logger.warning(
                    f"No propositions generated for chunk {index + 1} "
                    f"(content: {chunk.page_content[:100]}...)"
                )
                continue

            # Filter out empty or whitespace-only propositions
            valid_propositions = [p for p in propositions_text if p and p.strip()]
            if len(valid_propositions) < len(propositions_text):
                logger.debug(
                    f"Filtered out {len(propositions_text) - len(valid_propositions)} "
                    f"empty propositions from chunk {index + 1}"
                )

            # Create Document objects for each valid proposition
            for prop_text in valid_propositions:
                prop_metadata = chunk.metadata.copy()
                prop_doc = Document(
                    page_content=prop_text.strip(), metadata=prop_metadata
                )
                all_propositions.append(prop_doc)

            # Update progress
            if (index + 1) % max(1, total_chunks // 10) == 0 or (
                index + 1
            ) == total_chunks:
                progress_pct = ((index + 1) / total_chunks) * 100
                if progress_callback:
                    progress_callback(progress_pct)

            logger.debug(
                f"Generated {len(valid_propositions)} valid propositions "
                f"(from {len(propositions_text)} total) from chunk {index + 1}"
            )

        logger.info(
            f"Generated {len(all_propositions)} total valid propositions "
            f"from {total_chunks} chunks"
        )

        if not all_propositions:
            error_msg = (
                "No valid propositions were generated from any chunks. "
                "Cannot proceed with indexing."
            )
            logger.error(error_msg)
            raise ValueError(error_msg)

        # Quality check and filter propositions
        logger.debug(
            f"Starting quality filtering of {len(all_propositions)} propositions "
            f"with thresholds: {quality_thresholds}"
        )
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
            # Filter out empty documents even in fallback
            filtered_propositions = [
                prop
                for prop in all_propositions
                if prop.page_content and prop.page_content.strip()
            ]

            if not filtered_propositions:
                error_msg = (
                    f"All {len(all_propositions)} generated propositions were empty or invalid. "
                    "Cannot index document."
                )
                logger.error(error_msg)
                raise ValueError(
                    "No valid propositions generated. All propositions were empty or failed quality checks."
                )

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
            ValueError: If no valid (non-empty) documents to index
            Exception: If ChromaDB operation fails
        """
        # Filter out any empty documents (defensive check)
        valid_chunks = [
            chunk
            for chunk in chunks
            if chunk.page_content and chunk.page_content.strip()
        ]

        if not valid_chunks:
            error_msg = (
                f"No valid propositions to index for document {document_id}. "
                f"All {len(chunks)} propositions were empty."
            )
            logger.error(error_msg)
            raise ValueError(error_msg)

        if len(valid_chunks) < len(chunks):
            logger.warning(
                f"Filtered out {len(chunks) - len(valid_chunks)} empty propositions "
                f"before indexing {len(valid_chunks)} valid ones"
            )

        collection = get_chroma_collection(document_id)
        collection.add_documents(valid_chunks)
        logger.info(
            f"Indexed {len(valid_chunks)} propositions to ChromaDB for doc {document_id}"
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
