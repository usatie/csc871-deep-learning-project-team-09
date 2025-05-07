import os
from dataclasses import dataclass
from typing import Dict, List, Tuple, Callable


@dataclass
class TranslationConfig:
    # Language and tokenizer settings
    src_lang: str
    tgt_lang: str
    spacy_models: Dict[
        str, str
    ]  # e.g. {'zh': 'zh_core_web_sm', 'en': 'en_core_web_sm'}
    dataset_loader: Callable[[], Tuple]  # e.g. load() -> (train, val, test)

    # Vocabulary settings
    min_freq: int = 2
    specials: List[str] = ("<s>", "</s>", "<blank>", "<unk>")
    vocab_path: str = "vocab.pk"

    # Distributed training settings
    distributed: bool = False
    master_port: str = "12355"

    # Model architecture settings
    model_layers: int = 6
    d_model: int = 512
    d_ff: int = 2048
    h: int = 8
    dropout: float = 0.1
    smoothing: float = 0.1

    # Training hyperparameters
    batch_size: int = 32
    max_len: int = 72
    num_epochs: int = 8
    accum_iter: int = 10
    base_lr: float = 1.0
    warmup: int = 3000
    file_prefix: str = "model_"

    def __post_init__(self):
        # Ensure the output directory exists
        vocab_dir = os.path.dirname(self.vocab_path)
        if vocab_dir:  # Only create directory if path contains a directory
            os.makedirs(vocab_dir, exist_ok=True)


def get_default_config() -> TranslationConfig:
    """Get default configuration for training."""
    return TranslationConfig(
        src_lang="zh",
        tgt_lang="en",
        spacy_models={"zh": "zh_core_web_sm", "en": "en_core_web_sm"},
        dataset_loader=lambda: None,  # This should be set by the user
        min_freq=2,
        specials=["<s>", "</s>", "<blank>", "<unk>"],
        vocab_path="vocab.pk",
        distributed=False,
        master_port="12355",
        model_layers=6,
        d_model=512,
        d_ff=2048,
        h=8,
        dropout=0.1,
        smoothing=0.1,
        batch_size=32,
        max_len=72,
        num_epochs=8,
        accum_iter=10,
        base_lr=1.0,
        warmup=3000,
        file_prefix="model_",
    )


def get_checkpoint_dir(cfg: TranslationConfig) -> str:
    return os.path.join("checkpoints", cfg.file_prefix)


def get_final_checkpoint_path(cfg: TranslationConfig) -> str:
    return os.path.join(
        get_checkpoint_dir(cfg),
        f"final_bs{cfg.batch_size}_acc{cfg.accum_iter}_lr{cfg.base_lr}_warm{cfg.warmup}_ep{cfg.num_epochs}.pt",
    )


def get_checkpoint_path(cfg: TranslationConfig, epoch: int) -> str:
    return os.path.join(
        get_checkpoint_dir(cfg),
        f"epoch_{epoch:02d}_bs{cfg.batch_size}_acc{cfg.accum_iter}_lr{cfg.base_lr}_warm{cfg.warmup}_ep{cfg.num_epochs}.pt",
    )


def get_checkpoint_files(cfg: TranslationConfig) -> List[str]:
    checkpoint_dir = get_checkpoint_dir(cfg)
    checkpoint_files = [
        f
        for f in os.listdir(checkpoint_dir)
        if f.endswith(".pt") and f.startswith("epoch_")
    ]
    # Filter out files that don't match the current hyperparameters
    print(f"checkpoint_files: {checkpoint_files}")
    print(
        f"filter: {f'bs{cfg.batch_size}_acc{cfg.accum_iter}_lr{cfg.base_lr}_warm{cfg.warmup}_ep{cfg.num_epochs}.pt'}"
    )
    checkpoint_files = [
        f
        for f in checkpoint_files
        if f.endswith(
            f"bs{cfg.batch_size}_acc{cfg.accum_iter}_lr{cfg.base_lr}_warm{cfg.warmup}_ep{cfg.num_epochs}.pt"
        )
    ]
    return checkpoint_files


def print_config(cfg: TranslationConfig):
    """Print all configuration parameters in a readable format."""
    print(f"\n{'='*50}")
    print(f"CONFIGURATION SUMMARY:")
    print(f"{'='*50}")

    # Language and tokenizer settings
    print(f"LANGUAGE SETTINGS:")
    print(f"  Source language: {cfg.src_lang}")
    print(f"  Target language: {cfg.tgt_lang}")
    print(f"  Spacy models: {cfg.spacy_models}")

    # Vocabulary settings
    print(f"\nVOCABULARY SETTINGS:")
    print(f"  Minimum frequency: {cfg.min_freq}")
    print(f"  Special tokens: {', '.join(cfg.specials)}")
    print(f"  Vocabulary path: {cfg.vocab_path}")

    # Distributed training settings
    print(f"\nDISTRIBUTED TRAINING:")
    print(f"  Distributed: {'Enabled' if cfg.distributed else 'Disabled'}")
    if cfg.distributed:
        print(f"  Master port: {cfg.master_port}")

    # Model architecture settings
    print(f"\nMODEL ARCHITECTURE:")
    print(f"  Layers: {cfg.model_layers}")
    print(f"  Dimension (d_model): {cfg.d_model}")
    print(f"  Feed-forward dimension: {cfg.d_ff}")
    print(f"  Attention heads: {cfg.h}")
    print(f"  Dropout rate: {cfg.dropout}")
    print(f"  Label smoothing: {cfg.smoothing}")

    # Training hyperparameters
    print(f"\nTRAINING HYPERPARAMETERS:")
    print(f"  Batch size: {cfg.batch_size}")
    print(f"  Maximum sequence length: {cfg.max_len}")
    print(f"  Number of epochs: {cfg.num_epochs}")
    print(f"  Gradient accumulation steps: {cfg.accum_iter}")
    print(f"  Base learning rate: {cfg.base_lr}")
    print(f"  Warmup steps: {cfg.warmup}")
    print(f"  Model file prefix: {cfg.file_prefix}")

    print(f"{'='*50}")
