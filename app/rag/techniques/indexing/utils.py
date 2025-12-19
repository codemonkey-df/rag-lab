"""
Shared utilities for document ingestion.

Common functions used across multiple ingestion strategies.
"""

import logging
from typing import List

from langchain_community.document_loaders import PyMuPDFLoader
from langchain_core.documents import Document

logger = logging.getLogger(__name__)


def load_pdf_with_metadata(file_path: str) -> List[Document]:
    """
    Load PDF using LangChain PyMuPDFLoader with page/line metadata.

    Args:
        file_path: Path to the PDF file

    Returns:
        List of Document objects with metadata:
        - page: page number
        - source: file path
        - (line numbers added during chunking)
    """
    loader = PyMuPDFLoader(file_path)
    documents = loader.load()

    # Ensure metadata includes page numbers
    for i, doc in enumerate(documents):
        if "page" not in doc.metadata:
            doc.metadata["page"] = i + 1
        if "source" not in doc.metadata:
            doc.metadata["source"] = file_path

    logger.info(f"Loaded PDF: {len(documents)} pages from {file_path}")
    return documents
