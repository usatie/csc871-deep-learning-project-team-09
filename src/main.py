import torch
import argparse
from transformer.config.config import TranslationConfig, get_default_config
from transformer.train import train_model
from transformer.data.dataset import multi30k_loader
# Need to check if these dataset loaders exist
from transformer.data.dataset import tatoeba_zh_en_loader

def get_config(dataset):
    if dataset == "multi30k":
        return multi30k_config()
    elif dataset == "tatoeba_zh_en":
        return tatoeba_zh_en_config()
    elif dataset == "tatoeba_ja_en":
        return tatoeba_ja_en_config()
    else:
        raise ValueError(f"Unknown dataset: {dataset}")

def multi30k_config():
    # Example usage with Multi30k dataset
    cfg = TranslationConfig(
        src_lang="de",
        tgt_lang="en",
        spacy_models={
            "de": "de_core_news_sm",
            "en": "en_core_web_sm"
        },
        vocab_path='vocab_multi30k_de_en.pk',
        dataset_loader=multi30k_loader,
        distributed=torch.cuda.device_count() > 1,
        file_prefix='multi30k_model_',
        batch_size=32,
    )
    return cfg

def tatoeba_zh_en_config():
    # Configuration for Chinese-English dataset
    cfg = TranslationConfig(
        src_lang="zh",
        tgt_lang="en",
        spacy_models={
            "zh": "zh_core_web_sm",
            "en": "en_core_web_sm"
        },
        vocab_path='vocab_tatoeba_zh_en.pk',
        dataset_loader=tatoeba_zh_en_loader,
        distributed=torch.cuda.device_count() > 1,
        file_prefix='tatoeba_zh_en_model_',
        batch_size=32,
    )
    return cfg

def tatoeba_ja_en_config():
    # Configuration for Japanese-English dataset
    cfg = TranslationConfig(
        src_lang="ja",
        tgt_lang="en",
        spacy_models={
            "ja": "ja_core_news_sm",
            "en": "en_core_web_sm"
        },
        vocab_path='vocab_tatoeba_ja_en.pk',
        dataset_loader=None, # TODO: Add dataset loader
        distributed=torch.cuda.device_count() > 1,
        file_prefix='tatoeba_ja_en_model_',
        batch_size=32,
    )
    return cfg

def parse_args():
    parser = argparse.ArgumentParser(description='Transformer for translation')
    parser.add_argument('--dataset', type=str, default='multi30k', 
                        choices=['multi30k', 'tatoeba_zh_en', 'tatoeba_ja_en'],
                        help='Dataset to use for training')
    return parser.parse_args()

def main():
    args = parse_args()
    torch.cuda.set_per_process_memory_fraction(0.8)
    cfg = get_config(args.dataset)
    print(cfg)
    train_model(cfg)

if __name__ == "__main__":
    main() 