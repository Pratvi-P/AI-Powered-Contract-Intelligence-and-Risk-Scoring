"""
Day 8: FastAPI Backend

Endpoints:
  POST /analyze      - Upload a PDF, get full analysis back
  GET  /health       - Health check
  GET  /categories   - List all 41 clause categories the model can detect
  POST /search       - Search similar contracts by plain text query

Usage:
    uvicorn src.main:app --reload --host 0.0.0.0 --port 8000

Then open: http://localhost:8000/docs  (Swagger UI, free interactive testing)
"""

import json
import os
import tempfile
from pathlib import Path

from fastapi import FastAPI, File, HTTPException, Query, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from src.pipeline import ContractPipeline
from dotenv import load_dotenv

load_dotenv()
# ── App setup ──────────────────────────────────────────────────────────────
app = FastAPI(
    title="Contract Intelligence API",
    description="AI-powered legal contract analysis: clause detection, NER, semantic search.",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Load pipeline once at startup (not on every request) ──────────────────
CATEGORIES_PATH = Path("evaluation") / ".." / "data" / "categories.json"
THRESHOLDS_PATH = Path("evaluation") / "thresholds.json"

pipeline = ContractPipeline(
    thresholds_path="evaluation/thresholds.json",
    categories_path="evaluation/categories.json",
)


# ── Request / Response models ──────────────────────────────────────────────
class SearchRequest(BaseModel):
    query: str
    top_k: int = 5
    filter_clause: str = None


# ── Endpoints ──────────────────────────────────────────────────────────────
@app.get("/health")
def health():
    """Quick check that the API is running."""
    return {"status": "ok", "version": "1.0.0"}


@app.get("/categories")
def get_categories():
    """Return the list of 41 clause categories the model can detect."""
    return {
        "categories": pipeline.categories,
        "count": len(pipeline.categories),
    }


@app.post("/analyze")
async def analyze_contract(file: UploadFile = File(...)):
    """
    Upload a PDF contract and get back:
    - Extracted text (per page)
    - Named entities (organizations, dates, money, locations)
    - Detected clause types with confidence scores
    - Top 5 semantically similar contracts from the database
    """
    if not file.filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files are supported.")

    # Save upload to a temp file so PyMuPDF can read it
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
        content = await file.read()
        tmp.write(content)
        tmp_path = tmp.name

    try:
        result = pipeline.analyze(tmp_path, filename=file.filename)
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        os.unlink(tmp_path)  # clean up temp file


@app.post("/search")
def search_contracts(request: SearchRequest):
    """
    Search for similar contract clauses using plain English.
    Optionally filter by clause type.

    Example:
        {"query": "governing law California", "filter_clause": "Governing Law"}
    """
    try:
        results = pipeline.search(
            query=request.query,
            top_k=request.top_k,
            filter_clause=request.filter_clause,
        )
        return {"query": request.query, "results": results}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
