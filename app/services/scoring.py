"""
Scoring and metrics calculation
"""
from sklearn.metrics.pairwise import cosine_similarity
from app.core.dependencies import get_embeddings
import numpy as np
from typing import List


async def calculate_semantic_variance(response1: str, response2: str) -> float:
    """
    Calculate cosine similarity between two responses.
    
    Args:
        response1: First response text
        response2: Second response text
        
    Returns:
        Cosine similarity score (0.0 to 1.0)
    """
    embeddings = get_embeddings()
    
    # Embed both responses
    emb1 = await embeddings.aembed_query(response1)
    emb2 = await embeddings.aembed_query(response2)
    
    # Calculate cosine similarity
    similarity = cosine_similarity([emb1], [emb2])[0][0]
    
    return float(similarity)


async def calculate_batch_semantic_variance(responses: List[str]) -> np.ndarray:
    """
    Calculate pairwise semantic variance for multiple responses.
    
    More efficient than calculating individual pairs when comparing
    multiple results.
    
    Args:
        responses: List of response texts to compare
        
    Returns:
        Numpy array (similarity matrix) where element [i][j] is the
        cosine similarity between response i and response j
    """
    embeddings = get_embeddings()
    
    # Embed all responses
    embedded = []
    for response in responses:
        emb = await embeddings.aembed_query(response)
        embedded.append(emb)
    
    # Calculate pairwise cosine similarity
    similarity_matrix = cosine_similarity(embedded)
    
    return similarity_matrix
