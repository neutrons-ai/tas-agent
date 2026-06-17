### Vector Store
import numpy as np
import chromadb
import hashlib
import os
from typing import List, Any

class VectorStore:
    """Manage document embeddings in a chromaDB vector store.

    Accepts documents of any source type (pdf, txt, web, ...). The
    ``file_type`` recorded in each document's metadata is preserved so
    collections can be filtered by source later.
    """
    def __init__(self, collection_name: str = "documents", persistent_directory: str = "../data/vector_store"):
        self.collection_name = collection_name
        self.persistent_directory = persistent_directory
        self.client = None
        self.collection = None
        self._initialize_store()

    def _initialize_store(self):
        try:
            os.makedirs(self.persistent_directory, exist_ok =True)
            self.client = chromadb.PersistentClient(path = self.persistent_directory)
            self.collection = self.client.get_or_create_collection(
                name = self.collection_name,
                metadata = {
                    "description": "Document embeddings for RAG (pdf, txt, web)",
                    # Use cosine distance so similarity = 1 - distance is valid.
                    # NOTE: the metric is fixed at creation; call reset_collection()
                    # to rebuild an existing collection that used another metric.
                    "hnsw:space": "cosine",
                },
            )
            print(f"Vector store initialied. Collection: {self.collection_name}")
            print(f"Existing documents in collection: {self.collection.count()}")
        except Exception as e:
            raise ValueError(e)
        
    def reset_collection(self):
        """Drop and recreate the collection.

        Use this when an existing collection was created with a different
        distance metric (e.g. the default L2) and needs to be rebuilt with
        the cosine space configured in ``_initialize_store``. After calling
        this, re-run ingestion to repopulate the collection.
        """
        print(f"Resetting collection: {self.collection_name}")
        try:
            self.client.delete_collection(self.collection_name)
        except Exception:
            # Collection may not exist yet; that's fine.
            pass
        self._initialize_store()

    def add_documents(self, documents: List[Any], embeddings: np.ndarray):
        if len(documents) != len(embeddings):
            raise ValueError("Number of documents must match number of embeddings")
        print(f"Adding {len(documents)} documents to vector store....")

        ids = []
        metadatas = []
        documents_text = []
        embeddings_list = []
        seen_ids = set()  # dedup within this batch

        for i, (doc, embedding) in enumerate(zip(documents, embeddings)):
            # Deterministic id from content so identical chunks collapse to one
            # row, making re-ingestion idempotent (no duplicate pile-up).
            doc_id = "doc_" + hashlib.sha1(doc.page_content.encode("utf-8")).hexdigest()[:16]
            if doc_id in seen_ids:
                continue
            seen_ids.add(doc_id)

            metadata = self._sanitize_metadata(doc.metadata)
            metadata["doc_index"] = i
            metadata["content_length"] = len(doc.page_content)
            metadata.setdefault("file_type", "unknown")

            ids.append(doc_id)
            metadatas.append(metadata)
            documents_text.append(doc.page_content)
            embeddings_list.append(embedding.tolist())

        # Skip ids already stored from a previous run.
        try:
            existing = set(self.collection.get(ids=ids).get("ids", []))
        except Exception:
            existing = set()

        new_idx = [i for i, doc_id in enumerate(ids) if doc_id not in existing]
        skipped = len(documents) - len(new_idx)
        if not new_idx:
            print(f"  No new documents to add ({skipped} duplicates skipped).")
            return

        try:
            self.collection.add(
                ids = [ids[i] for i in new_idx],
                embeddings = [embeddings_list[i] for i in new_idx],
                metadatas=[metadatas[i] for i in new_idx],
                documents=[documents_text[i] for i in new_idx],
            )
            print(f"  Added {len(new_idx)} documents ({skipped} duplicates skipped).")
        except Exception as e:
            raise ValueError(e)

    @staticmethod
    def _sanitize_metadata(metadata: dict) -> dict:
        """Coerce metadata into Chroma-acceptable scalar values.

        Chroma only accepts str/int/float/bool metadata values. PDF/web/txt
        loaders attach assorted types (lists, dicts, None), so drop None and
        stringify anything that isn't already a scalar.
        """
        clean = {}
        for key, value in dict(metadata).items():
            if value is None:
                continue
            if isinstance(value, (str, int, float, bool)):
                clean[key] = value
            else:
                clean[key] = str(value)
        return clean