### Retriever pipeline from VectorStore
from typing import Dict, Any
from vector_store import VectorStore
from embedding import EmbeddingManager

class RAGRetriever:
    """Handles query-based retrieval from the vector store."""
    def __init__(self, vector_store: VectorStore, embedding_manager: EmbeddingManager):
        """Init."""
        self.vector_store = vector_store
        self.embedding_manager = embedding_manager
    
    def retrieve(self, query: str, top_k: int = 5, score_threshold: float = 0.0,
                 where: Dict[str, Any] = None) -> list[Dict[str, Any]]:
        """Retrieve relevant documents for a query.

        ``where``: optional Chroma metadata filter (e.g.
        {"source_file": "https://.../publications"}) to scope retrieval to a
        single page. Useful for enumeration queries ("list all X") where the
        answer is spread across many chunks of one source.
        """
        print(f"Retrieving documents for query: '{query}'")
        print(f"Top K: {top_k}, score threshold: {score_threshold}, where: {where}")

        # Generate query embedding
        query_embedding = self.embedding_manager.generate_embeddings([query])[0]

        try:
            query_kwargs = {
                "query_embeddings": [query_embedding.tolist()],
                "n_results": top_k,
            }
            if where:
                query_kwargs["where"] = where
            results = self.vector_store.collection.query(**query_kwargs)
            retrieved_docs = []
            if results['documents'] and results['documents'][0]:
                documents = results['documents'][0]
                metadatas = results['metadatas'][0]
                distances = results['distances'][0]
                ids = results['ids'][0]

                for i, (doc_id, document, metadata, distance) in enumerate(zip(ids, documents, metadatas, distances)):
                    # convert distance to similarity score (ChromaDB uses cosine distance)
                    similarity_score = 1 - distance

                    if similarity_score >= score_threshold:
                        retrieved_docs.append({
                            "id": doc_id,
                            "content": document,
                            "metadata": metadata,
                            "similarity_score": similarity_score,
                            "distance": distance,
                            "rank": i + 1
                        })
                print(f"Retrieved {len(retrieved_docs)} documents (after filtering)")
            else:
                print("No documents found.")
            return retrieved_docs
        except Exception as e:
            print(e)  
            return []
