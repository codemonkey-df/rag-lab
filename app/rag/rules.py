"""
RAG technique combination rules.

Rules define what combinations are allowed based on the three-layer architecture:
- Layer 1: Indexing strategies (mutually exclusive)
- Layer 2: Query pipeline components (mix & match)
- Layer 3: Advanced controllers (mutually exclusive, max one)

Rules:
- Layer 3 (Advanced Controllers):
  * SelfRAG/CRAG: Can use basic_rag or fusion_retrieval (depends on implementation)
  * AdaptiveRetrieval: Always uses fusion_retrieval (hardcoded)
- Standard pipelines: More combinations allowed (HyDE, reranking, compression, etc.)
"""

from typing import List

from app.models.enums import (
    LAYER_1_TECHNIQUES,
    LAYER_2_TECHNIQUES,
    LAYER_3_TECHNIQUES,
    RAGTechnique,
)


class TechniqueValidator:
    """Validates three-layer technique combinations according to RAG rules."""

    def validate(
        self, techniques: List[RAGTechnique]
    ) -> tuple[bool, List[str], List[str]]:
        """
        Validate technique combination.

        Returns:
            (is_valid, errors, warnings)
        """
        errors = []
        warnings = []

        # Separate by layer
        layer1 = [t for t in techniques if t in LAYER_1_TECHNIQUES]
        layer2 = [t for t in techniques if t in LAYER_2_TECHNIQUES]
        layer3 = [t for t in techniques if t in LAYER_3_TECHNIQUES]

        # Layer 1: Must have exactly ONE (or none for query-time)
        if len(layer1) > 1:
            errors.append(
                f"Layer 1 (Indexing): Can only select ONE technique. Found: {[t.value for t in layer1]}"
            )

        # Layer 1: Mutually exclusive checks
        if (
            RAGTechnique.SEMANTIC_CHUNKING in layer1
            and RAGTechnique.PROPOSITION_CHUNKING in layer1
        ):
            errors.append(
                "Cannot combine Semantic Chunking + Proposition Chunking (different storage structures)"
            )

        if RAGTechnique.PARENT_DOCUMENT in layer1 and len(layer1) > 1:
            errors.append(
                "Parent Document is mutually exclusive with other indexing strategies"
            )

        if (
            RAGTechnique.PROPOSITION_CHUNKING in layer1
            and RAGTechnique.CONTEXTUAL_HEADERS in layer1
        ):
            errors.append("Proposition Chunking cannot combine with Contextual Headers")

        # Layer 2: Must have EITHER Basic RAG OR Fusion (not both)
        if RAGTechnique.BASIC_RAG in layer2 and RAGTechnique.FUSION_RETRIEVAL in layer2:
            errors.append(
                "Cannot combine Basic RAG + Fusion Retrieval (Fusion contains Basic)"
            )

        # Layer 2: Reranking requires retrieval
        if RAGTechnique.RERANKING in layer2:
            if (
                RAGTechnique.BASIC_RAG not in layer2
                and RAGTechnique.FUSION_RETRIEVAL not in layer2
            ):
                errors.append(
                    "Reranking requires a retrieval technique (Basic RAG or Fusion Retrieval)"
                )

        # Layer 2: Compression requires retrieval
        if RAGTechnique.CONTEXTUAL_COMPRESSION in layer2:
            if (
                RAGTechnique.BASIC_RAG not in layer2
                and RAGTechnique.FUSION_RETRIEVAL not in layer2
            ):
                errors.append(
                    "Contextual Compression requires a retrieval technique (Basic RAG or Fusion Retrieval)"
                )

        # Layer 3: Max ONE
        if len(layer3) > 1:
            errors.append(
                f"Layer 3 (Advanced): Can select MAX ONE technique. Found: {[t.value for t in layer3]}"
            )

        # Layer 3: Self-RAG + CRAG conflict
        if RAGTechnique.SELF_RAG in layer3 and RAGTechnique.CRAG in layer3:
            errors.append(
                "Cannot combine Self-RAG + CRAG (logic collision - both have critique loops)"
            )

        # Layer 3: CRAG/Self-RAG can only use Basic RAG or Fusion Retrieval
        if RAGTechnique.CRAG in layer3 or RAGTechnique.SELF_RAG in layer3:
            allowed_layer2 = {RAGTechnique.BASIC_RAG, RAGTechnique.FUSION_RETRIEVAL}
            incompatible_layer2 = [
                t for t in layer2
                if t not in allowed_layer2
            ]
            if incompatible_layer2:
                incompatible_names = [t.value.replace("_", " ").title() for t in incompatible_layer2]
                technique_name = "CRAG" if RAGTechnique.CRAG in layer3 else "Self-RAG"
                errors.append(
                    f"{technique_name} can only use Basic RAG or Fusion Retrieval. "
                    f"Other Layer 2 techniques ({', '.join(incompatible_names)}) are not supported."
                )

        # Layer 3: Adaptive Retrieval is standalone
        if RAGTechnique.ADAPTIVE_RETRIEVAL in layer3:
            if layer2:
                layer2_names = [t.value.replace("_", " ").title() for t in layer2]
                errors.append(
                    f"Adaptive Retrieval is a standalone technique and cannot be combined with any Layer 2 techniques. "
                    f"Found: {', '.join(layer2_names)}"
                )

        # Warnings for slow techniques
        if RAGTechnique.SEMANTIC_CHUNKING in layer1:
            warnings.append(
                "Semantic Chunking: Slower than standard (embeds during split)"
            )

        if RAGTechnique.CONTEXTUAL_HEADERS in layer1:
            warnings.append(
                "Contextual Headers: Very slow (~30 min for 20-page PDF on local LLM)"
            )

        if RAGTechnique.PROPOSITION_CHUNKING in layer1:
            warnings.append(
                "Proposition Chunking: Extremely slow (LLM rewrites every sentence)"
            )

        if RAGTechnique.SELF_RAG in layer3:
            warnings.append("Self-RAG: 3x latency (3 LLM calls per query)")

        if RAGTechnique.CRAG in layer3:
            warnings.append("CRAG: Requires web search, may increase latency")

        is_valid = len(errors) == 0
        return is_valid, errors, warnings
