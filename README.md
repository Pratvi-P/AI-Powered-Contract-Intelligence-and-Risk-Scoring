# AI-Powered Contract Intelligence & Risk Scoring

An end-to-end NLP system that automatically analyzes legal contracts — extracting entities, detecting clause types, and enabling semantic search across a contract database.

---

## What it does

Upload any contract PDF and the system returns:
- **Named entities** — company names, dates, monetary values, locations
- **Clause detection** — which of 41 legal clause types are present (Governing Law, Termination, Audit Rights, etc.)
- **Semantic search** — top 5 most similar contracts from a 32,000-paragraph database

---

## Architecture

```
PDF Upload
    ↓
extract_pdf_text.py     → plain text extraction (PyMuPDF)
    ↓
run_ner.py              → named entity recognition (spaCy)
    ↓
Legal-BERT model        → clause classification (fine-tuned transformer)
    ↓
Pinecone vector DB      → semantic similarity search
    ↓
FastAPI (/analyze)      → single JSON response
```

---

## Project Structure

```
├── src/
│   ├── main.py             # FastAPI app — all endpoints
│   ├── pipeline.py         # ties all components together
│   ├── classifier.py       # clause classification
│   └── search.py           # Pinecone semantic search
├── evaluation/
│   ├── per_category_results.json   # per-category F1 scores
│   ├── thresholds.json             # tuned confidence thresholds
│   └── categories.json             # 41 clause category names
├── vector_db/
│   └── vector_db_config.json       # Pinecone index config
├── prepare_data.py         # downloads + processes CUAD dataset
├── extract_pdf_text.py     # PDF text extraction pipeline
├── run_ner.py              # NER on contract text
├── finetune.ipynb          # Legal-BERT fine-tuning (Colab)
├── day6_evaluation.ipynb   # model evaluation + threshold tuning (Colab)
├── day7_vectordb.ipynb     # embeddings + Pinecone upload (Colab)
├── load_test.py            # Locust load testing
├── Dockerfile              # container definition
├── docker-compose.yml      # container orchestration
├── requirements.txt        # ML/data dependencies
├── requirements_api.txt    # API dependencies
└── .env.example            # environment variables template
```

---

## Setup & Running

### 1. Clone the repo
```bash
git clone https://github.com/<your-username>/AI-Powered-Contract-Intelligence-and-Risk-Scoring.git
cd AI-Powered-Contract-Intelligence-and-Risk-Scoring
```

### 2. Create virtual environment
```bash
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
pip install -r requirements.txt
pip install -r requirements_api.txt
python -m spacy download en_core_web_sm
```

### 3. Set environment variables
```bash
cp .env.example .env
# Edit .env and add your Pinecone API key
```

### 4. Start the API
```bash
uvicorn src.main:app --reload --host 0.0.0.0 --port 8000
```

### 5. Open Swagger UI
```
http://localhost:8000/docs
```

---

## API Endpoints

| Method | Endpoint | Description |
|---|---|---|
| GET | `/health` | Health check |
| GET | `/categories` | List all 41 clause categories |
| POST | `/analyze` | Upload PDF, get full analysis |
| POST | `/search` | Semantic contract search |

---

## Model Performance

Trained on the CUAD dataset (510 legal contracts, 41 clause categories).

| Metric | Default Threshold | Tuned Threshold |
|---|---|---|
| F1 Micro | 0.670 | **0.697** |
| Precision | 0.787 | 0.755 |
| Recall | 0.583 | 0.647 |

**Best performing categories:**
- Governing Law: F1 = 0.923
- Insurance: F1 = 0.878
- Parties: F1 = 0.843

**Weak categories** (limited training examples):
- Most Favored Nation, Source Code Escrow, Third Party Beneficiary

---

## Technology Stack

| Component | Technology |
|---|---|
| NLP Model | Legal-BERT (nlpaueb/legal-bert-base-uncased) |
| NER | spaCy en_core_web_sm |
| PDF Extraction | PyMuPDF |
| Vector Database | Pinecone |
| Embeddings | all-MiniLM-L6-v2 |
| API Framework | FastAPI + Uvicorn |
| Training Dataset | CUAD (Contract Understanding Atticus Dataset) |
| Training Platform | Google Colab (T4 GPU) |
| Containerization | Docker |

---

## Known Limitations

- Classifier runs in mock mode without HF_TOKEN and HF_MODEL_URL configured
- OCR for scanned PDFs requires Tesseract binary (not installed by default)
- 9 clause categories score below F1=0.3 due to limited training examples
- Model loaded from Hugging Face Hub (requires internet connection)

---

## Future Improvements

- Fine-tune on more examples for weak categories
- Add per-category threshold tuning at inference time
- Build a frontend UI for non-technical users
- Add async inference with Celery for large documents
- Set up monitoring and logging (Prometheus + Grafana)