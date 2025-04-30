import argparse
import sys
from pathlib import Path

# Add src to Python path
src_path = str(Path(__file__).parent.parent)
if src_path not in sys.path:
    sys.path.append(src_path)

from transformer.evaluation import run_model_example
from transformer.config.dataset_configs import get_config


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
        "--num_examples", type=int, default=3, help="Number of examples to translate"
    )
    return parser.parse_args()


def main():
    args = parse_args()
    cfg = get_config(args.dataset)
    # Override some settings for inference
    cfg.distributed = False  # Not needed for inference
    cfg.batch_size = 1  # Single example at a time
    print(cfg)

    run_model_example(cfg, args.num_examples)


if __name__ == "__main__":
    main()
