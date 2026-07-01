"""
Semantic search over contract chunks using Pinecone.

Requires:
    PINECONE_API_KEY  in .env
    vector_db/vector_db_config.json  (saved by Day 7 notebook)
"""

import json
import os
from pathlib import Path


class ContractSearcher:
    """Wraps Pinecone index for semantic contract search."""

    def __init__(self, config_path: str = "vector_db/vector_db_config.json"):
        self.ready = False

        api_key = os.getenv("PINECONE_API_KEY")
        if not api_key:
            print(
                "WARNING: PINECONE_API_KEY not set. "
                "Search endpoint will return empty results. "
                "Add it to your .env file."
            )
            return

        try:
            from pinecone import Pinecone
            from sentence_transformers import SentenceTransformer

            with open(config_path) as f:
                config = json.load(f)

            self.index_name     = config["pinecone_index_name"]
            self.embedding_dim  = config["embedding_dim"]
            self.model_name     = config["embedding_model"]

            print(f"Loading embedding model: {self.model_name} ...")
            self.embedder = SentenceTransformer(self.model_name)

            print(f"Connecting to Pinecone index: {self.index_name} ...")
            pc          = Pinecone(api_key=api_key)
            self.index  = pc.Index(self.index_name)
            self.ready  = True
            print("Search ready.")

        except ImportError as e:
            print(f"WARNING: Missing dependency for search: {e}. "
                  "Run: pip install pinecone sentence-transformers")
        except Exception as e:
            print(f"WARNING: Could not initialize search: {e}")

    def search(
        self,
        query: str,
        top_k: int = 5,
        filter_clause: str = None,
    ) -> list[dict]:
        """
        Search for contract chunks semantically similar to the query.

        Args:
            query:         Plain English query
            top_k:         Number of results to return
            filter_clause: Optional clause type filter
                           e.g. "Governing Law"
        """
        if not self.ready:
            return [{
                "note": "Search not available. Set PINECONE_API_KEY in .env."
            }]

        query_vec = self.embedder.encode(
            query, normalize_embeddings=True
        ).tolist()

        filter_dict = None
        if filter_clause:
            filter_dict = {"clauses": {"$in": [filter_clause]}}

        raw = self.index.query(
            vector=query_vec,
            top_k=top_k,
            include_metadata=True,
            filter=filter_dict,
        )

        results = []
        for match in raw["matches"]:
            meta = match["metadata"]
            results.append({
                "score":       round(match["score"], 4),
                "contract_id": meta.get("contract_id", ""),
                "clauses":     meta.get("clauses", []),
                "text":        meta.get("text", "")[:300],
            })

        return results
