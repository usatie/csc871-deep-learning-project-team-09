import argparse

from transformer.evaluation import run_model_example
from transformer.config import get_config, print_config


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run translation examples using trained transformer model"
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default="multi30k",
        choices=["multi30k", "tatoeba_zh_en", "tatoeba_ja_en"],
        help="Dataset to use for translation",
    )
    parser.add_argument(
        "--num-examples", type=int, default=3, help="Number of examples to translate"
    )
    parser.add_argument(
        "--num-epochs", type=int, default=8, help="Number of epochs to train for"
    )
    return parser.parse_args()


def main():
    args = parse_args()
    cfg = get_config(args.dataset)
    # Override some settings for inference
    cfg.distributed = False  # Not needed for inference
    cfg.num_epochs = args.num_epochs
    print_config(cfg)

    run_model_example(cfg, args.num_examples)


if __name__ == "__main__":
    main()
