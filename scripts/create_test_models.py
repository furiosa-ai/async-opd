#!/usr/bin/env python3
"""Create tiny test models for integration tests.

Generates two small Qwen3 models with random weights (different seeds)
for use as student and teacher in integration tests. Same architecture,
same vocab, different weights → non-zero KL for all loss modes.

Uses 1 layer + small hidden dims. Embedding dominates model size (~9.7M
of 9.8M params) due to Qwen3's 151k vocab, but load time is still fast
(<2s vs ~5s for the full 0.6B model).

Usage:
    python scripts/create_test_models.py

Output:
    tests/fixtures/tiny_student/  (~40MB, 1-layer Qwen3, seed=42)
    tests/fixtures/tiny_teacher/  (~40MB, 1-layer Qwen3, seed=123)
"""

from pathlib import Path

import torch
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

PROJECT_ROOT = Path(__file__).resolve().parent.parent
FIXTURES = PROJECT_ROOT / "tests" / "fixtures"
STUDENT_PATH = FIXTURES / "tiny_student"
TEACHER_PATH = FIXTURES / "tiny_teacher"

# Base config from Qwen3-0.6B (same architecture used in production configs)
BASE_MODEL = "Qwen/Qwen3-0.6B"
NUM_LAYERS = 1
OVERRIDES = dict(
    num_hidden_layers=NUM_LAYERS,
    hidden_size=64,
    intermediate_size=128,
    num_attention_heads=4,
    num_key_value_heads=2,
    # Keep original vocab_size — special tokens (eos=151645, pad=151643) require it.
    # Qwen3 requires layer_types to match num_hidden_layers
    layer_types=["full_attention"] * NUM_LAYERS,
)


def create_model(seed, output_path):
    """Create a tiny model with random weights from the given seed."""
    config = AutoConfig.from_pretrained(BASE_MODEL, trust_remote_code=True)
    for k, v in OVERRIDES.items():
        setattr(config, k, v)

    torch.manual_seed(seed)
    model = AutoModelForCausalLM.from_config(config, trust_remote_code=True)

    n_params = sum(p.numel() for p in model.parameters())
    print(f"  Created model: {n_params/1e6:.1f}M params, seed={seed}")

    output_path.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(output_path)

    # Copy tokenizer from base model
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL, trust_remote_code=True)
    tokenizer.save_pretrained(output_path)

    print(f"  Saved to {output_path}")
    return n_params


def main():
    print("Creating tiny test models...")
    print(f"Base: {BASE_MODEL}")
    print(f"Overrides: {OVERRIDES}\n")

    print("Student:")
    n_student = create_model(seed=42, output_path=STUDENT_PATH)

    print("\nTeacher:")
    n_teacher = create_model(seed=123, output_path=TEACHER_PATH)

    print(f"\nDone. Student: {n_student/1e6:.1f}M, Teacher: {n_teacher/1e6:.1f}M")
    print(f"Paths:\n  {STUDENT_PATH}\n  {TEACHER_PATH}")


if __name__ == "__main__":
    main()
