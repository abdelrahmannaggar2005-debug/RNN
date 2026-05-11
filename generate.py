"""
generate.py
===========
Text Generation — Load trained model and sample from it.

Usage:
    python generate.py                                 # default seed + settings
    python generate.py --seed "To be or not"           # custom seed
    python generate.py --temp 0.5 --length 300         # custom temperature
    python generate.py --compare                       # compare all temperatures
"""

import numpy as np
import sys, os, argparse

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from data.text_loader import CharDataset, load_text
from models.language_model import LanguageModel


WEIGHT_PATH = 'saved_weights/lm_weights.npz'
CELL_TYPE   = 'lstm'
EMBED_DIM   = 32
HIDDEN_DIM  = 128


def build_model_and_dataset():
    text    = load_text(data_dir='data')
    dataset = CharDataset(text, seq_len=50)
    model   = LanguageModel(dataset.vocab_size, EMBED_DIM, HIDDEN_DIM, CELL_TYPE)
    if not os.path.exists(WEIGHT_PATH):
        print(f"[ERROR] No saved weights at {WEIGHT_PATH}. Run train.py first.")
        sys.exit(1)
    model.load(WEIGHT_PATH)
    return model, dataset


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed',    default='First Citizen:\n', help='Seed text')
    parser.add_argument('--temp',    type=float, default=0.8,  help='Temperature')
    parser.add_argument('--length',  type=int,   default=300,  help='Chars to generate')
    parser.add_argument('--compare', action='store_true',       help='Show multiple temps')
    args = parser.parse_args()

    model, dataset = build_model_and_dataset()

    if args.compare:
        print("\n══ Temperature Comparison ══════════════════════════════\n")
        for T in [0.3, 0.7, 1.0, 1.3, 1.8]:
            text = model.generate(args.seed, dataset.char2idx, dataset.idx2char,
                                  length=args.length, temperature=T)
            print(f"── Temperature {T} ──────────────────────────────────")
            print(text)
            print()
    else:
        text = model.generate(args.seed, dataset.char2idx, dataset.idx2char,
                              length=args.length, temperature=args.temp)
        print("\n══ Generated Text ══════════════════════════════════════\n")
        print(text)


if __name__ == '__main__':
    main()
