

import argparse
import json
import os
import re
import zipfile
from collections import defaultdict
from urllib.request import urlretrieve

from sklearn.model_selection import train_test_split

CUAD_ZIP_URL = "https://github.com/TheAtticusProject/cuad/raw/main/data.zip"


def get_category_name(question: str) -> str:

    match = re.search(r'"([^"]+)"', question)
    if match:
        return match.group(1).strip()
    return question.strip()[:60]


def build_chunks(context: str, chunk_words: int, stride_words: int):

    words = [(m.start(), m.end()) for m in re.finditer(r"\S+", context)]
    if not words:
        return []

    chunks = []
    i = 0
    while i < len(words):
        j = min(i + chunk_words, len(words))
        start_char = words[i][0]
        end_char = words[j - 1][1]
        chunks.append((start_char, end_char, context[start_char:end_char]))
        if j == len(words):
            break
        i += stride_words
    return chunks


def download_cuad(raw_cache_dir: str) -> str:
    os.makedirs(raw_cache_dir, exist_ok=True)
    json_path = os.path.join(raw_cache_dir, "CUADv1.json")

    if os.path.exists(json_path):
        print(f"Found cached {json_path}, skipping download.")
        return json_path

    zip_path = os.path.join(raw_cache_dir, "_cuad_data.zip")
    print(f"Downloading CUAD v1 from {CUAD_ZIP_URL} (~18MB, one-time)...")
    urlretrieve(CUAD_ZIP_URL, zip_path)

    print("Extracting CUADv1.json...")
    with zipfile.ZipFile(zip_path) as zf:
        zf.extract("CUADv1.json", raw_cache_dir)

    os.remove(zip_path)
    return json_path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_dir", default="./data")
    parser.add_argument("--raw_cache_dir", default="./data/_raw",
                         help="Where the downloaded CUADv1.json is cached")
    parser.add_argument("--chunk_words", type=int, default=200,
                         help="Approx words per chunk (a rough proxy for tokens)")
    parser.add_argument("--stride_words", type=int, default=100,
                         help="Word stride between chunks (creates overlap so "
                              "clauses near a chunk boundary aren't lost)")
    parser.add_argument("--train_frac", type=float, default=0.8)
    parser.add_argument("--val_frac", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    json_path = download_cuad(args.raw_cache_dir)
    print(f"Loading {json_path}...")
    with open(json_path, encoding="utf-8") as f:
        raw = json.load(f)


    contracts = {}  # title -> {"context": str, "spans": {category: [(start,end),...]}}
    categories = set()

    for entry in raw["data"]:
        title = entry["title"]
        for para in entry["paragraphs"]:
            context = para["context"]
            if title not in contracts:
                contracts[title] = {"context": context, "spans": defaultdict(list)}

            for qa in para["qas"]:
                category = get_category_name(qa["question"])
                categories.add(category)
                for ans in qa.get("answers", []):
                    text = ans.get("text", "")
                    start = ans.get("answer_start")
                    if text and start is not None:
                        contracts[title]["spans"][category].append(
                            (start, start + len(text))
                        )

    categories = sorted(categories)
    cat_to_idx = {c: i for i, c in enumerate(categories)}
    print(f"Found {len(contracts)} unique contracts and "
          f"{len(categories)} clause categories.")

    with open(os.path.join(args.output_dir, "categories.json"), "w") as f:
        json.dump(categories, f, indent=2)

    # Build chunk-level multi-label examples
    examples = []
    for title, info in contracts.items():
        context = info["context"]
        spans_by_cat = info["spans"]
        for start_char, end_char, chunk_text in build_chunks(
            context, args.chunk_words, args.stride_words
        ):
            if not chunk_text.strip():
                continue
            labels = [0] * len(categories)
            for cat, spans in spans_by_cat.items():
                for (s, e) in spans:
                    if s < end_char and e > start_char:  # overlap test
                        labels[cat_to_idx[cat]] = 1
            examples.append({
                "contract_id": title,
                "text": chunk_text,
                "labels": labels,
            })

    n_positive = sum(1 for ex in examples if any(ex["labels"]))
    print(f"Built {len(examples)} text chunks total "
          f"({n_positive} contain at least one clause, "
          f"{len(examples) - n_positive} are 'no clause' background text).")

    # Split by CONTRACT to avoid leaking the same contract's text across splits
    contract_ids = sorted(contracts.keys())
    train_ids, temp_ids = train_test_split(
        contract_ids, train_size=args.train_frac, random_state=args.seed
    )
    val_frac_of_temp = args.val_frac / (1 - args.train_frac)
    val_ids, test_ids = train_test_split(
        temp_ids, train_size=val_frac_of_temp, random_state=args.seed
    )
    train_ids, val_ids, test_ids = set(train_ids), set(val_ids), set(test_ids)

    splits = {"train": [], "validation": [], "test": []}
    for ex in examples:
        if ex["contract_id"] in train_ids:
            splits["train"].append(ex)
        elif ex["contract_id"] in val_ids:
            splits["validation"].append(ex)
        else:
            splits["test"].append(ex)

    for split_name, split_data in splits.items():
        out_path = os.path.join(args.output_dir, f"{split_name}.jsonl")
        with open(out_path, "w") as f:
            for ex in split_data:
                f.write(json.dumps(ex) + "\n")
        print(f"  {split_name:<10}: {len(split_data)} examples -> {out_path}")

    print("\nDone. Dataset ready for train_classifier.py")


if __name__ == "__main__":
    main()