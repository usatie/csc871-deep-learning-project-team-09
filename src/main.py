import argparse
import sys

def parse_args():
    parser = argparse.ArgumentParser(description='Transformer translation system')
    parser.add_argument('mode', type=str, choices=['train', 'run'],
                        help='Mode to run: train or run examples')
    return parser.parse_args()

def main():
    args = parse_args()
    
    if args.mode == 'train':
        # Import and run training
        from train import main as train_main
        train_main()
    else:
        # Import and run examples
        from run_example import main as run_main
        run_main()

if __name__ == "__main__":
    main() 