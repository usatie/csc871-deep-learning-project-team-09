import torch
from transformer.config.config import TranslationConfig, get_default_config
from transformer.train import train_model
from transformer.data.dataset import multi30k_loader

def main():
    torch.cuda.set_per_process_memory_fraction(0.8)
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
    print(cfg)
    print(f"Initial GPU memory allocated: {torch.cuda.memory_allocated() / 1e6:.2f} MB")
    print(f"Initial GPU memory reserved: {torch.cuda.memory_reserved() / 1e6:.2f} MB")
    train_model(cfg)

if __name__ == "__main__":
    main() 