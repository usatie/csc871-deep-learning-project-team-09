from dataclasses import dataclass
from typing import Dict, List, Tuple, Callable
import os

@dataclass
class TranslationConfig:
    # Language and tokenizer settings
    src_lang: str
    tgt_lang: str
    spacy_models: Dict[str, str]  # e.g. {'zh': 'zh_core_web_sm', 'en': 'en_core_web_sm'}
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
        spacy_models={
            "zh": "zh_core_web_sm",
            "en": "en_core_web_sm"
        },
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
        file_prefix="model_"
    ) 