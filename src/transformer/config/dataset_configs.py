import torch

from transformer.config.config import TranslationConfig
from transformer.data.dataset import multi30k_loader, tatoeba_zh_en_loader


def multi30k_config():
    """Configuration for Multi30k German-English dataset."""
    return TranslationConfig(
        src_lang="de",
        tgt_lang="en",
        spacy_models={"de": "de_core_news_sm", "en": "en_core_web_sm"},
        vocab_path="vocab_multi30k_de_en.pk",
        dataset_loader=multi30k_loader,
        distributed=torch.cuda.device_count() > 1,
        file_prefix="multi30k_de_en",
        batch_size=32,
    )


def tatoeba_zh_en_config():
    """Configuration for Tatoeba Chinese-English dataset."""
    return TranslationConfig(
        src_lang="zh",
        tgt_lang="en",
        spacy_models={"zh": "zh_core_web_sm", "en": "en_core_web_sm"},
        vocab_path="vocab_tatoeba_zh_en.pk",
        dataset_loader=tatoeba_zh_en_loader,
        distributed=torch.cuda.device_count() > 1,
        file_prefix="tatoeba_zh_en",
        batch_size=32,
    )


def tatoeba_ja_en_config():
    """Configuration for Tatoeba Japanese-English dataset."""
    return TranslationConfig(
        src_lang="ja",
        tgt_lang="en",
        spacy_models={"ja": "ja_core_news_sm", "en": "en_core_web_sm"},
        vocab_path="vocab_tatoeba_ja_en.pk",
        dataset_loader=None,  # TODO: Add dataset loader
        distributed=torch.cuda.device_count() > 1,
        file_prefix="tatoeba_ja_en",
        batch_size=32,
    )


def get_config(dataset: str) -> TranslationConfig:
    """Get configuration for the specified dataset.

    Args:
        dataset: Name of the dataset ('multi30k', 'tatoeba_zh_en', or 'tatoeba_ja_en')

    Returns:
        TranslationConfig: Configuration for the specified dataset

    Raises:
        ValueError: If the dataset is not supported
    """
    if dataset == "multi30k":
        return multi30k_config()
    elif dataset == "tatoeba_zh_en":
        return tatoeba_zh_en_config()
    elif dataset == "tatoeba_ja_en":
        return tatoeba_ja_en_config()
    else:
        raise ValueError(f"Unknown dataset: {dataset}")
