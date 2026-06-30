
import json
import os
from typing import Optional

import requests


class ClauseClassifier:
    """
    Calls the Hugging Face Inference API to classify contract text.
    Falls back to a mock response if HF_TOKEN or HF_MODEL_URL is not set
    so the API still starts and runs during development.
    """

    def __init__(self, thresholds_path: str, categories_path: str):
        self.hf_token = os.getenv("HF_TOKEN")
        self.model_url = os.getenv("HF_MODEL_URL")  # set after uploading model to HF

        # Load per-category thresholds from Day 6
        with open(thresholds_path) as f:
            self.thresholds = json.load(f)

        # Load category names
        with open(categories_path) as f:
            self.categories = json.load(f)

        if not self.hf_token or not self.model_url:
            print(
                "WARNING: HF_TOKEN or HF_MODEL_URL not set. "
                "Classifier will return mock results. "
                "Set these in your .env file after uploading the model to HF Hub."
            )

    def classify(self, text: str) -> list[dict]:
        """
        Classify a text chunk and return detected clauses with scores.

        Returns list of:
            {"clause": "Governing Law", "score": 0.92, "detected": True}
        """
        if not self.hf_token or not self.model_url:
            return self._mock_classify(text)

        try:
            response = requests.post(
                self.model_url,
                headers={"Authorization": f"Bearer {self.hf_token}"},
                json={"inputs": text},
                timeout=30,
            )
            response.raise_for_status()
            raw = response.json()

            # HF multi-label classification returns list of {label, score}
            results = []
            scores_by_label = {item["label"]: item["score"] for item in raw[0]}
            for cat in self.categories:
                score = scores_by_label.get(cat, 0.0)
                threshold = self.thresholds.get(cat, 0.5)
                results.append({
                    "clause": cat,
                    "score": round(score, 4),
                    "detected": score >= threshold,
                })
            return results

        except Exception as e:
            print(f"HF inference error: {e}. Returning mock result.")
            return self._mock_classify(text)

    def classify_document(self, chunks: list[str]) -> dict:
        """
        Classify a full document by running each chunk through the model
        and aggregating results — a clause is detected if ANY chunk flags it.
        Returns the max score per category across all chunks.
        """
        if not chunks:
            return {}

        # Aggregate max score per category across all chunks
        max_scores = {cat: 0.0 for cat in self.categories}

        for chunk in chunks:
            chunk_results = self.classify(chunk)
            for item in chunk_results:
                cat = item["clause"]
                if item["score"] > max_scores[cat]:
                    max_scores[cat] = item["score"]

        # Apply per-category thresholds
        detected = []
        for cat in self.categories:
            score = max_scores[cat]
            threshold = self.thresholds.get(cat, 0.5)
            if score >= threshold:
                detected.append({
                    "clause": cat,
                    "score": round(score, 4),
                    "threshold_used": threshold,
                })

        return {
            "detected_clauses": sorted(detected, key=lambda x: -x["score"]),
            "total_detected": len(detected),
        }

    def _mock_classify(self, text: str) -> list[dict]:
        """
        Returns a placeholder result when HF credentials aren't configured.
        Used during local development so the API doesn't crash.
        """
        text_lower = text.lower()
        mock_hits = []
        keyword_map = {
            "Governing Law":               ["govern", "jurisdiction", "law of"],
            "Termination For Convenience": ["terminat", "convenience"],
            "Audit Rights":                ["audit", "inspect", "records"],
            "Expiration Date":             ["expir", "expire", "end date"],
            "Non-Compete":                 ["non-compete", "not compete", "competition"],
        }
        for clause, keywords in keyword_map.items():
            if any(kw in text_lower for kw in keywords):
                mock_hits.append({
                    "clause": clause,
                    "score": 0.85,
                    "detected": True,
                })
        return mock_hits if mock_hits else [
            {"clause": "mock_mode", "score": 0.0,
             "detected": False,
             "note": "Set HF_TOKEN and HF_MODEL_URL in .env to enable real classification"}
        ]
