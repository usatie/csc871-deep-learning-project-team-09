import os
from typing import List, Tuple

import torch

from .config.config import TranslationConfig
from .data.dataset import create_dataloaders, load_tokenizers
from .training.util import greedy_decode
from .training.trainer import load_trained_model


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
    _, valid_loader, _ = create_dataloaders(
        cfg,
        tokenizers,
        vocab_src,
        vocab_tgt,
        device=device,
    )
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
    final_checkpoint = os.path.join(
        "checkpoints", cfg.file_prefix, f"epoch_{cfg.num_epochs-1:02d}.pt"
    )
    if os.path.exists(final_checkpoint):
        print(f"Found existing checkpoint at {final_checkpoint}, skipping training")
        model, vocab_src, vocab_tgt = load_trained_model(cfg, final_checkpoint)
    else:
        print("No existing checkpoint found, starting training")
        raise ValueError("No existing checkpoint found, starting training")
    tokenizers = load_tokenizers(cfg.spacy_models)
    model.eval()
    print("Checking model outputs:")
    example_data = check_outputs(
        cfg, model, tokenizers, vocab_src, vocab_tgt, n_examples=n_examples
    )
    return model, example_data
