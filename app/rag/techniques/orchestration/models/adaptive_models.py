"""
Pydantic models for Adaptive RAG structured outputs.

These models define the structure for LLM responses in the Adaptive RAG pipeline,
providing type safety and validation.
"""

from typing import List, Literal

from pydantic import BaseModel, Field


class QueryCategory(BaseModel):
    """Query category classification."""

    category: Literal["Factual", "Analytical", "Opinion", "Contextual"] = Field(
        description="The category of the query: Factual, Analytical, Opinion, or Contextual",
        example="Factual",
    )


class RelevanceScore(BaseModel):
    """Relevance score for a document."""

    score: float = Field(
        description="The relevance score of the document to the query (1-10 scale)",
        ge=1.0,
        le=10.0,
        example=8.0,
    )


class SubQueries(BaseModel):
    """Generated sub-queries for comprehensive analysis."""

    sub_queries: List[str] = Field(
        description="List of sub-queries for comprehensive analysis",
        example=["What is the population of New York?", "What is the GDP of New York?"],
    )


class SelectedIndices(BaseModel):
    """Indices of selected documents."""

    indices: List[int] = Field(
        description="Indices of selected documents",
        example=[0, 1, 2, 3],
    )


class Viewpoints(BaseModel):
    """Distinct viewpoints on a topic."""

    viewpoints: List[str] = Field(
        description="List of distinct viewpoints or perspectives on the topic",
        example=["Viewpoint 1: ...", "Viewpoint 2: ..."],
    )
