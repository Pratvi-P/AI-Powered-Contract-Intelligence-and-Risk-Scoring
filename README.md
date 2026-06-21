# Week 2, Day 1-4: Transformer Fine-Tuning on CUAD

Fine-tunes a pre-trained transformer (Legal-BERT / RoBERTa) to classify
which legal clause categories appear in a contract paragraph. Deliverable:
a fine-tuned clause classification model.

## How CUAD was adapted

CUAD is natively a **question-answering** dataset: 41 clause categories,
each phrased as a question, with answer spans inside full contract texts.
To get a **classifier** (matching your timetable), `prepare_data.py`:

1. Downloads CUAD from the Hugging Face Hub.
2. Splits each contract into overlapping ~200-word chunks.
3. Labels each chunk with every category whose answer span falls inside it
   → a multi-label classification example (a chunk can match 0, 1, or
   several categories at once).
4. Splits chunks into train/validation/test **by contract**, so the same
   contract never appears in two splits.

## Setup

```bash
pip install -r requirements.txt
python check_env.py          # confirms GPU + library versions
```

## Step 1 — Prepare the data (Day 1-2)

```bash
python prepare_data.py --output_dir ./data
```

This downloads CUAD (~100MB, cached after first run) and writes:
- `data/train.jsonl`, `data/validation.jsonl`, `data/test.jsonl`
- `data/categories.json` — the 41 label names, in label order

Expect roughly an 80/10/10 contract split. Most chunks will have **no**
clause label (they're just background contract text) — that's expected;
CUAD's true clauses are sparse. You'll deal with this imbalance more
deliberately in the Day 5-7 evaluation task.

## Step 2 — Fine-tune (Day 3-4)

```bash
python train_classifier.py \
    --data_dir ./data \
    --model_name nlpaueb/legal-bert-base-uncased \
    --output_dir ./outputs/model \
    --epochs 3 \
    --batch_size 8
```

Outputs:
- `outputs/model/final/` — the fine-tuned model + tokenizer
- `outputs/model/test_metrics.json` — micro/macro F1, precision, recall

### Model choices (`--model_name`)
| Model | Notes |
|---|---|
| `nlpaueb/legal-bert-base-uncased` | Legal-domain BERT, recommended default |
| `roberta-base` | Strong general-purpose baseline |
| `saibo/legal-roberta-base` | Legal-domain RoBERTa |
| `distilbert-base-uncased` | ~2x faster, lower VRAM, slightly lower accuracy |

### If you hit GPU out-of-memory
- Lower `--batch_size` (e.g. 4) and raise `--grad_accum_steps` (e.g. 4) —
  same effective batch size, less memory.
- Lower `--max_length` (e.g. 128) — shorter chunks use less memory.
- Try `distilbert-base-uncased` instead of a full BERT/RoBERTa model.

### Expected runtime
On a single mid-range GPU (e.g. RTX 3060/4060, 8-12GB VRAM), expect
roughly 20-60 minutes for 3 epochs depending on dataset size and model.
On CPU only, this could take many hours — consider `--epochs 1` and a
smaller `--max_length` for a first sanity-check run.

## What's next (Day 5-7, not in this script)

Per your timetable, Day 5-7 covers **Model Evaluation & Heuristics**:
digging into per-category precision/recall (some of the 41 categories
have very few examples), tuning the classification threshold per
category instead of one global 0.5, and adding post-processing rules to
improve confidence scoring. This script intentionally stops at a solid
baseline fine-tuned model + basic test metrics so that work has
something concrete to evaluate.

## Files in this delivery

| File | Purpose |
|---|---|
| `check_env.py` | Verify GPU/library setup |
| `prepare_data.py` | Download CUAD, build the chunked classification dataset |
| `train_classifier.py` | Fine-tune the transformer, save model + metrics |
| `requirements.txt` | Python dependencies |
