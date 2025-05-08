import argparse
import torch

from transformer.config import (
    get_checkpoint_files,
    print_config,
    get_config,
)
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
    parser.add_argument(
        "--num-epochs",
        type=int,
        default=8,
        help="Number of epochs to train for (default: 8)",
    )
    # Training hyperparameters
    parser.add_argument(
        "--batch-size",
        type=int,
        help="Batch size (default: from config)",
    )
    parser.add_argument(
        "--accum-iter",
        type=int,
        help="Gradient accumulation steps (default: from config)",
    )
    parser.add_argument(
        "--warmup",
        type=int,
        help="Number of warmup steps (default: from config)",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    cfg = get_config(args.dataset)
    # Override number of epochs from command line
    cfg.num_epochs = args.num_epochs
    cfg.distributed = torch.cuda.device_count() > 1

    # Override other hyperparameters if specified
    if args.batch_size is not None:
        cfg.batch_size = args.batch_size
    if args.accum_iter is not None:
        cfg.accum_iter = args.accum_iter
    if args.warmup is not None:
        cfg.warmup = args.warmup

    print_config(cfg)

    # Check if final checkpoint exists
    checkpoint_files = get_checkpoint_files(cfg)
    # Print the list each on a new line
    print(f"checkpoint_files:\n\t{'\n\t'.join(checkpoint_files)}")
    final_checkpoint = next(
        filter(lambda x: f"epoch_{cfg.num_epochs:02d}" in x, checkpoint_files), None
    )
    if args.force:
        print(f"Training from scratch, --force flag provided")
        train_model(cfg)
    elif final_checkpoint is not None:
        print(f"Found existing checkpoint at {final_checkpoint}, skipping training")
    else:
        print(f"Start training...")
        train_model(cfg)


if __name__ == "__main__":
    main()
