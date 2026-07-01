"""
Contract analysis pipeline.

Ties together:
  1. PDF text extraction  (extract_pdf_text.py logic)
  2. NER                  (run_ner.py logic)
  3. Clause classification (src/classifier.py)
  4. Semantic search       (src/search.py)
"""

import json
import re
from collections import defaultdict
from pathlib import Path

import fitz  # PyMuPDF
import spacy

from src.classifier import ClauseClassifier
from src.search import ContractSearcher

MIN_CHARS_PER_PAGE = 40
CHUNK_WORDS        = 200
STRIDE_WORDS       = 100
RELEVANT_NER       = {"ORG", "DATE", "MONEY", "GPE", "PERSON", "LAW"}
SPACY_MODEL        = "en_core_web_sm"


class ContractPipeline:
    def __init__(self, thresholds_path: str, categories_path: str):
        with open(categories_path) as f:
            self.categories = json.load(f)

        print(f"Loading spaCy model: {SPACY_MODEL} ...")
        try:
            self.nlp = spacy.load(SPACY_MODEL)
        except OSError:
            raise RuntimeError(
                f"spaCy model '{SPACY_MODEL}' not found. "
                f"Run: python -m spacy download {SPACY_MODEL}"
            )

        self.classifier = ClauseClassifier(
            thresholds_path=thresholds_path,
            categories_path=categories_path,
        )

        self.searcher = ContractSearcher()
        print("Pipeline ready.")

    # ── Step 1: PDF extraction ─────────────────────────────────────────────
    def extract_text(self, pdf_path: str) -> dict:
        doc   = fitz.open(pdf_path)
        pages = []
        for i, page in enumerate(doc):
            text = page.get_text("text").strip()
            pages.append({
                "page_number":    i + 1,
                "char_count":     len(text),
                "likely_scanned": len(text) < MIN_CHARS_PER_PAGE,
                "text":           text,
            })
        doc.close()
        return {
            "num_pages":    len(pages),
            "total_chars":  sum(p["char_count"] for p in pages),
            "likely_scanned_pages": sum(1 for p in pages if p["likely_scanned"]),
            "full_text":    "\n\n".join(p["text"] for p in pages),
            "pages":        pages,
        }

    # ── Step 2: NER ────────────────────────────────────────────────────────
    def extract_entities(self, text: str) -> dict:
        max_chars = self.nlp.max_length - 100
        chunks    = [text[i:i + max_chars] for i in range(0, len(text), max_chars)]
        counts    = defaultdict(lambda: defaultdict(int))

        for chunk in chunks:
            for ent in self.nlp(chunk).ents:
                if ent.label_ in RELEVANT_NER:
                    counts[ent.label_][" ".join(ent.text.split())] += 1

        result = {}
        for label, c in counts.items():
            result[label] = [
                {"text": t, "count": n}
                for t, n in sorted(c.items(), key=lambda x: -x[1])
            ]
        return result

    # ── Step 3: Chunk text for classification ──────────────────────────────
    def chunk_text(self, text: str) -> list[str]:
        words  = [(m.start(), m.end()) for m in re.finditer(r"\S+", text)]
        chunks = []
        i = 0
        while i < len(words):
            j         = min(i + CHUNK_WORDS, len(words))
            start_c   = words[i][0]
            end_c     = words[j - 1][1]
            chunks.append(text[start_c:end_c])
            if j == len(words):
                break
            i += STRIDE_WORDS
        return chunks

    # ── Main entry point ────────────────────────────────────────────────────
    def analyze(self, pdf_path: str, filename: str = "") -> dict:
        # Step 1: Extract text
        extraction = self.extract_text(pdf_path)
        full_text  = extraction["full_text"]

        # Step 2: NER
        entities = self.extract_entities(full_text)

        # Step 3: Classify clauses
        chunks           = self.chunk_text(full_text)
        classification   = self.classifier.classify_document(chunks)

        # Step 4: Semantic search for similar contracts
        search_query = full_text[:500]  # use first 500 chars as query
        similar      = self.searcher.search(search_query, top_k=5)

        return {
            "filename":    filename,
            "extraction": {
                "num_pages":            extraction["num_pages"],
                "total_chars":          extraction["total_chars"],
                "likely_scanned_pages": extraction["likely_scanned_pages"],
            },
            "entities":       entities,
            "classification": classification,
            "similar_contracts": similar,
        }

    # ── Search passthrough ──────────────────────────────────────────────────
    def search(self, query: str, top_k: int = 5,
               filter_clause: str = None) -> list[dict]:
        return self.searcher.search(query, top_k=top_k,
                                    filter_clause=filter_clause)
