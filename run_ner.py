"""
Day 3: Named Entity Recognition (NER) on Contract Text

Uses spaCy's pre-trained English model to automatically extract key entities
from contract text:
  - ORG        -> company/organization names
  - DATE       -> dates, durations ("two years", "January 2025")
  - MONEY      -> monetary values ("$50,000", "USD 1 million")
  - GPE        -> countries, cities, states ("California", "India")
  - PERSON     -> individual names
  - LAW        -> references to legal documents/acts

This runs on the JSON files produced by extract_pdf_text.py (Day 2).
Each contract gets a matching output JSON with all found entities.

How it fits the project:
  PDF -> extract_pdf_text.py -> plain text -> run_ner.py -> entities JSON
  The entities become searchable metadata in the FastAPI app (Day 8).
  E.g. "show me all contracts mentioning more than $1M" or
  "find contracts expiring before 2026" become possible.

Usage:
    # First time only - download the spaCy model:
    python -m spacy download en_core_web_trf

    # Then run NER on your extracted contracts:
    python run_ner.py --input_dir ./extracted_data --output_dir ./ner

Model choice:
    en_core_web_trf  -> transformer-based, most accurate, recommended
    en_core_web_lg   -> large, good accuracy, faster than trf
    en_core_web_sm   -> small/fast, lower accuracy (use only for quick tests)
"""

import argparse
import json
import os
from collections import defaultdict

import spacy


# Entity types we care about for legal contracts
# (spaCy detects more, but these are the most useful ones here)
RELEVANT_LABELS = {"ORG", "DATE", "MONEY", "GPE", "PERSON", "LAW"}


def load_model(model_name: str):
    """Load spaCy model with a helpful error message if not downloaded yet."""
    try:
        return spacy.load(model_name)
    except OSError:
        raise SystemExit(
            f"\nERROR: spaCy model '{model_name}' not found.\n"
            f"Download it by running:\n"
            f"    python -m spacy download {model_name}\n"
        )


def extract_entities(text: str, nlp) -> dict:
    """Run NER on a block of text and return structured entity results.

    spaCy has a max text length limit (the nlp.max_length attribute).
    Long contracts are processed in chunks to avoid hitting this limit.
    """
    max_chars = nlp.max_length - 100  # small safety margin
    chunks = [text[i:i + max_chars] for i in range(0, len(text), max_chars)]

    # entities_by_label: { "ORG": {"ABC Technologies": count, ...}, ... }
    entities_by_label = defaultdict(lambda: defaultdict(int))

    for chunk in chunks:
        doc = nlp(chunk)
        for ent in doc.ents:
            if ent.label_ in RELEVANT_LABELS:
                # Normalize whitespace but preserve case
                text_clean = " ".join(ent.text.split())
                entities_by_label[ent.label_][text_clean] += 1

    # Convert to sorted list of {text, count} per label, most frequent first
    result = {}
    for label, counts in entities_by_label.items():
        result[label] = [
            {"text": t, "count": c}
            for t, c in sorted(counts.items(), key=lambda x: -x[1])
        ]

    return result


def summarize_entities(entities: dict) -> dict:
    """Pull out the most useful summary fields for quick inspection."""
    summary = {}
    if "ORG" in entities:
        summary["organizations"] = [e["text"] for e in entities["ORG"][:10]]
    if "DATE" in entities:
        summary["dates"] = [e["text"] for e in entities["DATE"][:10]]
    if "MONEY" in entities:
        summary["monetary_values"] = [e["text"] for e in entities["MONEY"][:10]]
    if "GPE" in entities:
        summary["locations"] = [e["text"] for e in entities["GPE"][:10]]
    if "PERSON" in entities:
        summary["persons"] = [e["text"] for e in entities["PERSON"][:10]]
    if "LAW" in entities:
        summary["legal_references"] = [e["text"] for e in entities["LAW"][:10]]
    return summary


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_dir", default="./extracted_data",
                        help="Folder with JSON files from extract_pdf_text.py")
    parser.add_argument("--output_dir", default="./ner",
                        help="Where to save NER output JSONs")
    parser.add_argument("--model", default="en_core_web_trf",
                        help="spaCy model to use. Recommended: en_core_web_trf. "
                             "Fallback: en_core_web_sm (faster but less accurate)")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    print(f"Loading spaCy model: {args.model} ...")
    nlp = load_model(args.model)
    print("Model loaded.\n")

    json_files = [f for f in os.listdir(args.input_dir) if f.endswith(".json")]
    if not json_files:
        print(f"No JSON files found in {args.input_dir}")
        print("Run extract_pdf_text.py first to generate the input files.")
        return

    print(f"Found {len(json_files)} contract(s) to process.\n")

    for filename in json_files:
        in_path = os.path.join(args.input_dir, filename)
        print(f"Processing: {filename}")

        with open(in_path, encoding="utf-8") as f:
            contract = json.load(f)

        full_text = contract.get("full_text", "")
        if not full_text.strip():
            print(f"  WARNING: no text found in {filename}, skipping.")
            continue

        entities = extract_entities(full_text, nlp)
        summary = summarize_entities(entities)

        output = {
            "source_file": contract.get("source_file", filename),
            "num_pages": contract.get("num_pages"),
            "total_chars": contract.get("total_chars"),
            "entity_summary": summary,
            "all_entities": entities,
        }

        out_name = os.path.splitext(filename)[0] + "_ner.json"
        out_path = os.path.join(args.output_dir, out_name)
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(output, f, indent=2, ensure_ascii=False)

        # Print a quick preview to terminal
        print(f"  Organizations : {summary.get('organizations', [])[:4]}")
        print(f"  Dates         : {summary.get('dates', [])[:4]}")
        print(f"  Money         : {summary.get('monetary_values', [])[:4]}")
        print(f"  Locations     : {summary.get('locations', [])[:4]}")
        print(f"  Saved -> {out_path}\n")

    print("=" * 60)
    print("NER complete. Output files are in:", args.output_dir)
    print("Each file has:")
    print("  entity_summary  -> top entities per type (quick view)")
    print("  all_entities    -> every entity with frequency count")


if __name__ == "__main__":
    main()
