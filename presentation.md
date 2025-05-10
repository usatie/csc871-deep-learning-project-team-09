# Neural Machine Translation using Transformer
A Chinese–English Translation Case Study

---

## What is Machine Translation?

```
MAX_LEN = 128
BOS_TOK = "<s>"
EOS_TOK = "</s>"

def translate(model, src_tokens):
  next_tok = None
  output_tokens = [BOS_TOK]
  while next_tok != EOS_TOK and len(output_tokens) < MAX_LEN:
    next_tok = model.generate_next(src_tokens, output_tokens)
    output_tokens.append(next_tok)
  return output_tokens
```
*(This is a over-simplified greedy decoding approach)

---

## Motivation

1. We want to understand Transformer architecture in depth
2. We want to expand from CPU training to GPU training
3. We want to even explore distributed training on multiple GPUs

    -> Apply these concepts to a practical use case: Chinese-English translation

---

## Dataset

- Tatoeba Chinese-English Parallel Corpus (obtained April 27, 2025)
  - https://tatoeba.org/
- Number of sentence pairs: 71,155 total
- Number of unique source sentences: 60,260 total
- Multiple target translations per source sentence
    ![bg_down](https://cdn.markslides.ai/users/1557/images/ONBsLiPafYnUg29RRDGJH)
- Continuously being updated by users globally
- CC BY 2.0 License

---

## Dataset Details

- Split ratio: 80% train, 10% validation, 10% test
- Manually split by source sentence id
  
  | Split      | Source Sentence | Sentence Pairs       |
  |------------|-----------------|----------------------|
  | Training   | 48,208 (80.00%) | 56,830 pairs (79.87%)|
  | Validation | 6,026 (10.00%)  | 7,166 pairs (10.07%) |
  | Test       | 6,026 (10.00%)  | 7,159 pairs (10.06%) |

---

## Pre-processing
- Spacy tokenizers (https://spacy.io/)
  - zh_core_web_sm for Chinese, en_core_web_sm for English
- Max sequence length: 72 tokens
- Vocabulary sizes:
  - Source (Chinese): 15,465 tokens
  - Target (English): 9,733 tokens
  - Included words appearing at least 2 times in either train/val/test
  - 4 special tokens: `<s>`, `</s>`, `<unk>`, `<pad>`


---
# Implementation
---

## Implementation: Model Architecture

- Based on Vaswani et al. (2017) "Attention Is All You Need"
- Key differences from original paper (for the simplicity sake):
  - No shared embeddings between encoder and decoder
  - Using separate tokenizers for source and target
  - Using spacy's tokenizers instead of BPE
  - Greedy search instead of beam search
  - No checkpoint ensembling (using single best checkpoint)
---

## Architecture Details

- Encoder-Decoder structure with 6 layers each
- Embedding size: 512
- Feed-forward dimension: 2048
- Attention heads: 8
- Dropout rate: 0.1
- Total parameters: 62,034,949 (~237 MB)

---

## Architecture Components

- Embedding layers (separate for source/target)
- Multi-head Attention mechanisms
- Subsequent mask
- Positional Encoding 
- Feedforward layers
- Dropout and Layer Normalization

---

## Embedding Layer
- TODO

---

## Multi-head Attention
- TODO

---

## Subsequent mask for decoder
- TODO

---

## Positional Encoding 
- TODO

---

## Feedforward layers
- TODO

---

## Dropout and Layer Normalization
- TODO

---
# Training
---

## Training Overview

- Loss: CrossEntropy with label smoothing (0.1)
- Optimizer: Adam with Noam-style learning rate scheduler
- (Our implementation used 3000 warumup steps)

  ![](https://cdn.markslides.ai/users/1557/images/JPFj7bWzM9wzvQvH7vZJx)


---
## Learning Rate Scheduling

![bg right](https://images.unsplash.com/photo-1543286386-713bdd548da4?crop=entropy&cs=srgb&fm=jpg&ixid=M3w1NTU5NjN8MHwxfHNlYXJjaHwxfHxncmFwaHxlbnwwfHx8fDE3NDY4MzI2NTR8MA&ixlib=rb-4.1.0&q=85)
- Start from 0
- Gradually increase until the warmup period finishes
- Gradually decrease thereafter

---
## Potential issues of our implementation
- Our warmup (3000) was likely too large for the total training steps
  - Config 1 (BS=128): ~2200 accum steps after 100 epochs (136%)
  - Config 2 (BS=32): ~8900 accum steps after 50 epochs (34%)
- Original Paper:
  - 4000 warmup steps for total of 100,000 steps


---

## Label Smoothing


- Prevents overconfidence in model predictions
- Helps regularize the model
- Label smoothing = 0.1 used in our implementation

  | Label   |  class 1  |  class 2 | class 3|
  |------------|------------|------------|------------|
  | Hard Label | 0| 0|1 |
  | Smoothed Label | 0.05| 0.05|0.9|

---

## Training Configurations

- Two configurations tested:
  - Config 1: Batch size 128, Accumulation steps 21
  - Config 2: Batch size 32, Accumulation steps 10
- Tokens per weight update:
  - Config 1: ~2,688 tokens (128 * 21)
  - Config 2: ~320 tokens (32 * 10)
- Total weight updates:
  - Config 1: ~2,200 updates (100 epochs)
  - Config 2: ~8,900 updates (50 epochs)

---
# Distributed Training
---

## Distributed Training Overview

- PyTorch Distributed Data Parallel (DDP) framework
- Implementation steps:
  - Initialize process groups
  - Distribute data across GPUs
  - Reduce gradients during backward pass
  - Broadcast updated weights
  - Reduce values for logging

---

## Distributed Training Details

- Parallelized across 4 NVIDIA A100 GPUs on a single node
- Full model parameters kept on each GPU
- Gradient synchronization at backward pass
- Custom distributed training wrapper for easier debugging

---
# Evaluation
---

## Computational Platform

- Perlmutter supercomputer at NERSC
- Single GPU node configuration:
  - One AMD EPYC 7763 (Milan) processor
  - 64 cores with 204.8 GiB/s peak memory bandwidth
  - Four NVIDIA A100 (Ampere) GPUs

---

## Hardware Details

- GPU specifications:
  - 40GB HBM2 with 1555 GB/s memory bandwidth, or
  - 80GB HBM2e with 2039 GB/s memory bandwidth
- 3rd generation NVLink connections:
  - 4 links between each GPU pair
  - 25 GB/s/direction per link

---

## Training Results (Config 1)
<!-- TODO: Replace with loss curve -->
![bg right](https://images.unsplash.com/photo-1543286386-713bdd548da4?crop=entropy&cs=srgb&fm=jpg&ixid=M3w1NTU5NjN8MHwxfHNlYXJjaHwxfHxncmFwaHxlbnwwfHx8fDE3NDY4MzI2NTR8MA&ixlib=rb-4.1.0&q=85)

- Batch size 128, Accumulation steps 21
- Tokens/accum : approx. 25,000
- 100 epochs completed
- Gap between train/val loss indicates overfitting

---

## Evaluation: Training Results (Config 2)
<!-- TODO: Replace with loss curve -->
![bg right](https://images.unsplash.com/photo-1543286386-713bdd548da4?crop=entropy&cs=srgb&fm=jpg&ixid=M3w1NTU5NjN8MHwxfHNlYXJjaHwxfHxncmFwaHxlbnwwfHx8fDE3NDY4MzI2NTR8MA&ixlib=rb-4.1.0&q=85)


- Batch size 32, Accumulation steps 10
- Tokens/accum : approx. 4,000
- 54 epochs completed
  - Stopped early because it took very long and it was obvious that it was overfitting
- Gap between train/val loss indicates overfitting

---

## Example Translation 1 by early epoch model (epoch = 5)
- TODO: Add

---

## Example Translation 2 by mid epoch model (epoch = 50)
- TODO: Add

---

## Example Translation 3 by late epoch model (epoch = 100)
- TODO: Add


---

## Encoder Self Attention Visualization
- TODO: Add
---

## Decoder Self Attention Visualization
- TODO: Add
---

## Decoder Cross(src) Attention Visualization
- TODO: Add

---

## Overfitting Analysis

- Validation loss starts increasing:
  - Config 2 (BS=32): After epochs 10-15 (~1,500 updates)
  - Config 1 (BS=128): After epochs 40-60 (~1,000 updates)
- Larger batch size delayed overfitting in terms of epochs
- Similar overfitting point in terms of weight updates
- Possible reasons for overfitting:
  - Insufficient regularization
  - Limited dataset size
  - Short sentence pairs

---

## Training Performance Study

- Training speed comparison:
  - T4 GPU (Google Colab): [TBD] seconds/epoch
  - Single A100 (Perlmutter login node): [TBD] seconds/epoch
  - 4× A100 (Perlmutter GPU node):
    - Config 1 (BS=128): ~49 seconds/epoch
    - Config 2 (BS=32): ~135 seconds/epoch
- ~2.75× speedup with larger batch size

---

## Implementation Challenges

- Setting up distributed training environment on Perlmutter
- Data loading bottlenecks with large dataset
- Model convergence issues in early iterations
- GPU memory optimization for larger batch sizes
- Debugging cryptic DDP errors

---

## Solutions Implemented

- Custom distributed training wrapper around PyTorch DDP
- Optimized data loading with prefetching and pinned memory
- Learning rate warmup to stabilize early training
- Gradient accumulation to effectively increase batch size
- Detailed logging for distributed training debugging

---
# Conclusion
---

## Conclusion

- Successfully implemented full Transformer architecture for Chinese-English MT
- Demonstrated learning dynamics across different batch sizes
- Observed common seq2seq training patterns: fast initial learning, eventual overfitting
- Gained practical experience with distributed training on HPC

---

## Future Work

- Experiment with early stopping to prevent overfitting
- Try different regularization strategies
- Implement model compression techniques
- Implement beam search for better translation quality
- Implement tokenization using subword techniques (BPE, WordPiece)
- Train on larger datasets
- Weights share in embedding layer (encoder/decoder/generator)
- Explore multilingual models (zh -> multiple languages)

---

## Additional Directions

- Improve tokenization using subword techniques (BPE, WordPiece)
- Test on more diverse datasets beyond Tatoeba
- Explore multilingual models (zh → multiple languages)

---
# References
---

## References

- Vaswani et al., "Attention is All You Need", NeurIPS 2017
- The Annotated Transformer (Harvard NLP)
- PyTorch Distributed: https://pytorch.org/tutorials/intermediate/dist_tuto.html
- Tatoeba Project: https://tatoeba.org/
- NERSC Perlmutter: https://docs.nersc.gov/systems/perlmutter/architecture/

---
# Appendix
---
## Appendix: Model Size

```
Total parameters: 62,034,949
Total size: 236.64 MB

PARAMETER DISTRIBUTION:
  encoder layers: 18,914,304 params (30.6%)
  decoder layers: 25,224,192 params (40.8%)
  embeddings:     12,901,376 params (20.8%)
  generator:       4,993,029 params (8.0%)
  other:               2,048 params (0.0%)
```

---

## Appendix: Memory Usage

- Training memory footprint:
  - Model parameters: ~237 MB
  - Optimizer states: ~474 MB (Adam uses 2x params)
  - Activations: [TBD] MB per batch
  - Gradients: ~237 MB
- Total GPU memory usage:
  - Config 1 (BS=128): [TBD] GB
  - Config 2 (BS=32): [TBD] GB
