"""Pinned models and run constants (PLAN.md rev 3)."""
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

MODELS = {
    "qwen3-0.6b": ("Qwen/Qwen3-0.6B", "c1899de289a04d12100db370d81485cdf75e47ca"),
    "qwen3-1.7b": ("Qwen/Qwen3-1.7B", "70d244cc86ccca08cf5af4e1e306ecf908b1ad5e"),
}

MAX_NEW_TOKENS = 1024
TEMPERATURE = 0.6
TOP_P = 0.95
TOP_K = 20
NUM_THREADS = 4
BASE_SEED = 1234          # per-gen seed = BASE_SEED + hash-free index (see run.py)
NOCUE_SEEDS = (0, 1, 2)   # 3 samples for the no-cue cell; all other cells seed 0
ANSWER_MAX_TOKENS = 8     # tokens generated after forced '</think>' + answer prefix
