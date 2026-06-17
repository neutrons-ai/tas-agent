import numpy as np
from sentence_transformers import SentenceTransformer
import chromadb
from chromadb.config import Settings
import uuid
from typing import List, Dict, Any, Tuple
from sklearn.metrics.pairwise import cosine_similarity

class EmbeddingManager:
    """Hanldes document embedding generation using SentenceTransformer."""
    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        """Initialize embedding manager."""
        self.model_name = model_name
        self.model = None
        self._load_model() 
    
    def _load_model(self):
        try:
            print(f"Loading embeding model: {self.model_name}")
            self.model = SentenceTransformer(self.model_name)
            print(f"Model loaded successfully. Embedding dimension: {self.model.get_embedding_dimension()}")
        except Exception as e:
            print(f"Error loading model {self.model_name}: {e}")
            raise
    
    def generate_embeddings(self, texts: List[str]):
        if not self.model:
            raise ValueError("Model not loaded.")
        print(f"generating embedding for {len(texts)} texts...")
        embeddings = self.model.encode(texts, show_progress_bar=True)
        print(f"Generated embeddings with shape : {embeddings.shape}")
        return embeddings