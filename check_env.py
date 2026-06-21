"""
Week 2 - Day 1: Environment check.

Run this first to confirm your GPU and library versions are ready
for transformer fine-tuning.

Usage:
    python check_env.py
"""

import sys


def main():
    print("=" * 50)
    print("Environment Check")
    print("=" * 50)

    try:
        import torch
        print(f"PyTorch version : {torch.__version__}")
        cuda_ok = torch.cuda.is_available()
        print(f"CUDA available  : {cuda_ok}")
        if cuda_ok:
            print(f"  GPU            : {torch.cuda.get_device_name(0)}")
            vram_gb = torch.cuda.get_device_properties(0).total_memory / 1e9
            print(f"  VRAM           : {vram_gb:.1f} GB")
            if vram_gb < 6:
                print("  NOTE: <6GB VRAM. Use a small model (e.g. distilbert) "
                      "and/or batch_size=4 with gradient_accumulation_steps=4.")
        else:
            print("  WARNING: No GPU detected. Training will be slow on CPU. "
                  "Consider reducing dataset size / epochs for a first run.")
    except ImportError:
        print("PyTorch NOT installed. Run: pip install -r requirements.txt")
        sys.exit(1)

    try:
        import transformers
        print(f"Transformers    : {transformers.__version__}")
    except ImportError:
        print("transformers NOT installed.")
        sys.exit(1)

    try:
        import datasets
        print(f"Datasets        : {datasets.__version__}")
    except ImportError:
        print("datasets NOT installed.")
        sys.exit(1)

    try:
        import sklearn
        print(f"scikit-learn    : {sklearn.__version__}")
    except ImportError:
        print("scikit-learn NOT installed.")
        sys.exit(1)

    print("=" * 50)
    print("All core libraries found. You're ready for prepare_data.py")
    print("=" * 50)


if __name__ == "__main__":
    main()
