# Building and Evaluating a Lightweight Transformer for Japanese-English and Chinese-English Translation

## Overview
We aim to implement a simplified Transformer model from scratch and apply it to a Japanese-to-English (ja→en) and Chinese-to-English (ch→en) translation task using a small-scale dataset. Our focus is to better understand the internal mechanisms of Transformers (e.g., attention, positional encoding, masking) through hands-on implementation and experimentation, rather than achieving state-of-the-art performance.

## Project Structure
```
src/
├── __init__.py          # Makes src a Python package
├── main.py              # Main entry point
├── scripts/             # Command-line scripts
│   ├── run_example.py   # Script for running translation examples
│   └── train.py         # Training script
└── transformer/         # Core transformer implementation
    ├── config/          # Configuration files
    ├── data/            # Data loading and processing
    ├── evaluation.py    # Evaluation utilities
    ├── __init__.py      # Package exports
    ├── model/           # Model architecture
    ├── tests/           # Unit tests
    ├── training/        # Training-related code
    │   ├── trainer.py   # Core training logic
    │   └── __init__.py  # Package exports
    └── utils/           # Utility functions
```

## Installation

### Setting up the Environment
1. Create and activate the conda environment using the provided environment.yml:
```bash
conda env create -f environment.yml
conda activate transformer-translation
```

### Running the Translation Model
The script supports three datasets: Multi30k (German-English), Tatoeba Chinese-English, and Tatoeba Japanese-English.

#### Training the Model
To train the model:
```bash
# For Multi30k dataset (German-English)
python src/scripts/train.py --dataset multi30k

# For Chinese-English translation
python src/scripts/train.py --dataset tatoeba_zh_en

# For Japanese-English translation
python src/scripts/train.py --dataset tatoeba_ja_en

# Force retraining even if checkpoint exists
python src/scripts/train.py --dataset multi30k --force
```

#### Running Translation Examples
To run translation examples with a trained model:
```bash
# For Multi30k dataset (German-English)
python src/scripts/run_example.py --dataset multi30k --num_examples 5

# For Chinese-English translation
python src/scripts/run_example.py --dataset tatoeba_zh_en --num_examples 5

# For Japanese-English translation
python src/scripts/run_example.py --dataset tatoeba_ja_en --num_examples 5
```

The training script will:
1. Check for existing checkpoints in the `checkpoints` directory
2. If no checkpoint exists (or if --force is specified), train the model from scratch
3. Save checkpoints after each epoch

The example script will:
1. Load the trained model from the latest checkpoint
2. Run the specified number of translation examples
3. Display the source text, target text, and model's translation

## Dataset
We will use a subset of the Tatoeba corpus consisting of aligned Japanese-English and Chinese-English sentence pairs. The data will be filtered to include only short sentences (length < 50 tokens), and tokenization will be handled via a basic tokenizer or SentencePiece.

## Model Architecture
- Encoder–Decoder Transformer with 6 layers, 8 attention heads
- Positional encoding
- Masked multi-head self-attention in the decoder
- Teacher forcing during training

## Task & Goals
- Translate simple Japanese and Chinese sentences into English using a custom Transformer
- Visualize attention weights for interpretability
- Evaluate output quality using BLEU scores

## Team Member Roles
- Ruxue Jin: Transformer model implementation, positional encoding, masking
- Shun Usami: Dataset preparation, training pipeline, tokenization
- Yash Bhadiyadra: Evaluation (BLEU), attention visualization, report drafting

## Related Documents
- [Project Proposal](https://docs.google.com/document/d/10mLdkhueVMK_Vc57ZAEY3ufJZkFUEnUb9uTNaEkNaro/edit)
  - This document outlines the project goals, methodology, and expected outcomes. It serves as a reference for the project's scope and objectives.
- [Attention Is All You Need](https://arxiv.org/pdf/1706.03762)
  - This is the original paper introducing the Transformer architecture. It provides a comprehensive overview of the model's components, including self-attention, multi-head attention, and positional encoding.
- [BLEU: a Method for Automatic Evaluation of Machine Translation](https://dl.acm.org/doi/10.3115/1073083.1073135)
  - This paper introduces the BLEU score, a widely used metric for evaluating machine translation quality. It discusses the methodology behind BLEU and its effectiveness in comparing machine-generated translations to human references.
- [Illustrated Transformer](https://jalammar.github.io/illustrated-transformer/)
  - This blog post provides a visual and intuitive explanation of the Transformer architecture. It breaks down the components of the model, including self-attention, multi-head attention, and positional encoding, making it easier to understand how the Transformer works.
- [Annotated Transformer](http://nlp.seas.harvard.edu/annotated-transformer/)
  - This resource offers an annotated implementation of the Transformer model in PyTorch. It includes detailed explanations of each component, making it a valuable reference for understanding the inner workings of the Transformer architecture.
