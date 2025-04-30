import argparse
import os

from transformer.config.dataset_configs import get_config
from transformer.training.trainer import train_model


def parse_args():
    parser = argparse.ArgumentParser(
        description="Train transformer model for translation"
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default="multi30k",
        choices=["multi30k", "tatoeba_zh_en", "tatoeba_ja_en"],
        help="Dataset to use for training",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Force retraining even if checkpoint exists",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    cfg = get_config(args.dataset)
    print(cfg)

    # Check if final checkpoint exists
    final_checkpoint = os.path.join(
        "checkpoints", cfg.file_prefix, f"epoch_{cfg.num_epochs-1:02d}.pt"
    )
    if not os.path.exists(final_checkpoint) or args.force:
        print("Starting training...")
        train_model(cfg)
    else:
        print(f"Found existing checkpoint at {final_checkpoint}, skipping training")


if __name__ == "__main__":
    main()
