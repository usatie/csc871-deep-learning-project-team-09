import os
from typing import List, Tuple

import torch

from transformer.config import TranslationConfig, get_checkpoint_dir, get_checkpoint_files
from transformer.data.dataset import create_dataloaders, load_tokenizers
from transformer.training.util import greedy_decode
from transformer.training.trainer import load_trained_model

from .translation_examples import (
    check_outputs,
    run_model_example,
    print_data_stats,
)

__all__ = [
    "check_outputs",
    "run_model_example",
    "print_data_stats",
]


def check_outputs(
    cfg: TranslationConfig,
    model: torch.nn.Module,
    tokenizers,
    vocab_src,
    vocab_tgt,
    n_examples: int,
) -> List[Tuple]:
    """
    Check model outputs on validation data.

    Args:
        cfg: Translation configuration
        model: Trained model
        n_examples: Number of examples to check

    Returns:
        List of tuples containing (batch, src_tokens, tgt_tokens, output_ids, out_text, out_tokens)
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    train_loader, valid_loader, test_loader = create_dataloaders(
        cfg,
        tokenizers,
        vocab_src,
        vocab_tgt,
        device=device,
    )
    print_data_stats(cfg.batch_size, train_loader, valid_loader, test_loader)
    pad_id = vocab_tgt["<blank>"]
    eos_token = "</s>"
    results = []

    it = iter(valid_loader)
    for idx in range(n_examples):
        batch = next(it)

        src_tokens = [vocab_src.lookup_token(x) for x in batch.src[0] if x != pad_id]
        tgt_tokens = [vocab_tgt.lookup_token(x) for x in batch.tgt[0] if x != pad_id]

        print(f"\nExample {idx} ========")
        print("Source (input):  ", " ".join(src_tokens))
        print("Target (truth):  ", " ".join(tgt_tokens))

        # greedy decode
        output_ids = greedy_decode(
            model, batch.src, batch.src_mask, cfg.max_len, vocab_tgt["<s>"]
        )[0]
        out_tokens = [vocab_tgt.lookup_token(x) for x in output_ids if x != pad_id]
        if eos_token in out_tokens:
            out_tokens = out_tokens[: out_tokens.index(eos_token) + 1]
        out_text = " ".join(out_tokens)

        print("Model output:   ", out_text)
        results.append(
            (batch, src_tokens, tgt_tokens, output_ids, out_text, out_tokens)
        )

    return results


def run_model_example(
    cfg: TranslationConfig,
    n_examples: int,
) -> Tuple[torch.nn.Module, List[Tuple]]:
    """
    Load a trained model and run example translations.

    Args:
        cfg: Translation configuration
        n_examples: Number of examples to check

    Returns:
        Tuple of (model, example_data)
    """
    print("Loading model & vocab...")
    checkpoint_files = get_checkpoint_files(cfg)
    if not checkpoint_files:
        print("No existing checkpoint found, starting training")
        raise ValueError("No existing checkpoint found, starting training")
    final_checkpoint = os.path.join(get_checkpoint_dir(cfg), checkpoint_files[-1])
    print(f"Found existing checkpoint at {final_checkpoint}, skipping training")
    cfg.batch_size = 1  # We only need to run one example at a time

    model, vocab_src, vocab_tgt = load_trained_model(cfg, final_checkpoint)
    tokenizers = load_tokenizers(cfg.spacy_models)
    model.eval()
    print("Checking model outputs:")
    example_data = check_outputs(
        cfg, model, tokenizers, vocab_src, vocab_tgt, n_examples=n_examples
    )
    return model, example_data


def print_data_stats(
    batch_size, train_loader=None, valid_loader=None, test_loader=None
):
    """Print statistics about data loaders including batch counts and sample sizes."""
    print(f"\n{'='*50}")
    print(f"DATA SUMMARY:")
    print(f"{'='*50}")

    total_batches = 0

    if train_loader is not None:
        train_batches = len(train_loader)
        total_batches += train_batches
        train_samples = train_batches * batch_size
        print(f"Training:   {train_batches:4,} batches", end="")
        print(f" (~{train_samples:,} samples)")

    if valid_loader is not None:
        valid_batches = len(valid_loader)
        total_batches += valid_batches
        valid_samples = valid_batches * batch_size
        print(f"Validation: {valid_batches:4,} batches", end="")
        print(f" (~{valid_samples:,} samples)")

    if test_loader is not None:
        test_batches = len(test_loader)
        total_batches += test_batches
        test_samples = test_batches * batch_size
        print(f"Testing:    {test_batches:4,} batches", end="")
        print(f" (~{test_samples:,} samples)")

    print(f"{'-'*50}")
    print(f"Total:      {total_batches:4,} batches")
    print(f"Batch size: {batch_size}")
    print(f"Approx. total samples: {total_batches * batch_size:,}")

    print(f"{'='*50}")
