# Neural Machine Translation Using Transformer  
*A Chinese–English Translation Case Study*

<div style="text-align: left; font-size: 25px; margin-top: 30px;">
  👥 Team 9: <strong>Shun</strong>, <strong>Ruxue</strong>, <strong>Yash</strong>
</div>


---
## Motivation

- understand Transformer architecture in depth
- explore distributed training on multiple GPUs


---
## Overview

- Data Preparation
- Implementation
- Training Process
- Evaluation

---

## Dataset

- Tatoeba Chinese-English Parallel Corpus (obtained April 27, 2025)
- Number of sentence pairs: 71,155 total
<!-- - Number of unique source sentences: 60,260 total -->
- Multiple target translations per source sentence
    ![bg_down](https://cdn.markslides.ai/users/1557/images/ONBsLiPafYnUg29RRDGJH)
- Split ratio: 80% train, 10% validation, 10% test  
<!-- - Continuously being updated by users globally
- CC BY 2.0 License -->

<!--
footer: "https://tatoeba.org/"
-->



---

## Pre-processing
- Spacy tokenizers
  - zh_core_web_sm for Chinese
  - en_core_web_sm for English
- Vocabulary sizes:
  - Source (Chinese): 15,465 tokens
  - Target (English): 9,733 tokens
  - Included words appearing at least 2 times in either train/val/test
  - 4 special tokens: `<s>`, `</s>`, `<unk>`, `<pad>`
- Max sequence length: 72 tokens
<!--
footer: "https://arxiv.org/abs/1706.03762"
-->

---
# Comparison of Datasets and Methods



| | Team 9 | Attention is All you need |Ratio |
|:---|:---:|:---:|:---:|
| **source** | Tatoeba | WMT 2014 | - |
| **language** | Chinese — English | English — German| -|
| **Sentence Pairs** | 71,155 | 4.5 M |1:64|
| **Tokenize method** | SpaCy<br>zh_core_web_sm /<br>en_core_web_sm | BPE | -|
| **Vocabulary** | 15466(src) / 9733(tgt) | Shared 37,000 tokens|1:2.4|


---


# Implementation

---

## Implementation: Model Architecture

- Based on Vaswani et al. (2017) "Attention Is All You Need"
- Key differences from original paper (for the simplicity sake):
  - No shared embeddings between encoder and decoder
  - Using separate tokenizers for source and target
  - Greedy search instead of beam search
  - No checkpoint ensembling (using single best checkpoint)

<!--
footer: "https://arxiv.org/abs/1706.03762"
-->
---


## Architecture Details

- Encoder-Decoder structure with 6 layers each
- Embedding size: 512
- Feed-forward dimension: 2048
- Attention heads: 8
- Dropout rate: 0.1
- Total parameters: 62,034,949 (~237 MB)

---

## Architecture

<div style="display: flex; justify-content: space-between; align-items: center;">
  <div style="flex: 1;">
    <img src="https://cdn.markslides.ai/users/1657/images/cLwUK-0TGD7dGxinYQl4-" width="400px" alt="Left Image">
  </div>
  <div style="flex: 3;">
    <img src="https://cdn.markslides.ai/users/1657/images/n-75Am3WPaIBOSeDcfK0k" width="1000px" alt="Right Image">
  </div>
</div>

<!--
footer: "https://arxiv.org/abs/1706.03762"
-->


---
## EncoderLayer
<div style="display: flex; align-items: flex-start; gap: 20px;">
<!-- Left: Diagram -->
<div style="flex: 1; text-align: center; max-width: 50%;">
  <img src="https://cdn.markslides.ai/users/1657/images/WX6O4UBZFzg8opCI53pyc" width="100%" alt="Encoder Layer Diagram">
  <p style="font-size: 20px; margin-top: 10px;">
    input x: [batch_size, sequence_length, d_model] <br>
    output: [batch_size, sequence_length, d_model]
  </p>
</div>

<!-- Right: Code Blocks with Paragraphs -->
<div style="flex: 1; max-width: 50%; display: flex; flex-direction: column; gap: 10px;">
  <!-- First paragraph -->
  <p style="font-size: 16px; margin: 0 0 5px 0;">
    Apply residual connection to any sublayer with the same size:
  </p>
  
  <!-- First code block -->
  <pre style="background-color: #f6f8fa; padding: 12px; border-radius: 5px; margin: 0 0 15px 0;"><code style="font-family: 'Consolas', monospace; font-size: 16px; line-height: 1.4;">class SublayerConnection(nn.Module):
    def forward(self, x, sublayer):
        return x + self.dropout(sublayer(self.norm(x)))</code></pre>
        
  <!-- Second paragraph -->
  <p style="font-size: 16px; margin: 0 0 5px 0;">
    Follow Figure 1 (left) for connections:
  </p>
  
  <!-- Second code block -->
  <pre style="background-color: #f6f8fa; padding: 12px; border-radius: 5px; margin: 0;"><code style="font-family: 'Consolas', monospace; font-size: 16px; line-height: 1.4;">class EncoderLayer(nn.Module):
    def forward(self, x, mask):
        x = self.sublayer[0]
        (x, lambda x: self.self_attn(x, x, x, mask))
        return self.sublayer[1](x, self.feed_forward)</code></pre>
</div>
</div>



---

## DecoderLayer

<div style="display: flex; align-items: flex-start; gap: 20px;">
<!-- Left: Diagram -->
<div style="flex: 1; text-align: center; max-width: 50%;">
  <img src="https://cdn.markslides.ai/users/1657/images/rFPvLjP_jMYsfL0nkfrNQ" width="100%" alt="Encoder Layer Diagram">
  <p style="font-size: 20px; margin-top: 10px;">
    input x: [batch_size, sequence_length, d_model] <br>
    output: [batch_size, sequence_length, d_model]
  </p>
</div>

<!-- Right: Code Blocks with Paragraphs -->
<div style="flex: 1; max-width: 61%; display: flex; flex-direction: column; gap: 10px;">
  <!-- First paragraph -->
  <p style="font-size: 16px; margin: 0 0 5px 0;">
    Apply residual connection to any sublayer with the same size:
  </p>
  
  <!-- First code block -->
  <pre style="background-color: #f6f8fa; padding: 12px; border-radius: 5px; margin: 0 0 15px 0;"><code style="font-family: 'Consolas', monospace; font-size: 16px; line-height: 1.4;">class SublayerConnection(nn.Module):
    def forward(self, x, sublayer):
        return x + self.dropout(sublayer(self.norm(x)))</code></pre>
        
  <!-- Second paragraph -->
  <p style="font-size: 16px; margin: 0 0 5px 0;">
    Decoder is made of self-attn, src-attn, and feed forward (defined below)
  </p>
  
  <!-- Second code block -->
  <pre style="background-color: #f6f8fa; padding: 12px; border-radius: 5px; margin: 0;"><code style="font-family: 'Consolas', monospace; font-size: 16px; line-height: 1.4;">class DecoderLayer(nn.Module)
    def __init__(self, size, self_attn, src_attn, feed_forward, dropout):
        super(DecoderLayer, self).__init__()
        self.sublayer = clones(SublayerConnection(size, dropout), 3)
    def forward(self, x, memory, src_mask, tgt_mask):
        "Follow Figure 1 (right) for connections."
        m = memory
        x = self.sublayer[0](x, lambda x: self.self_attn(x, x, x, tgt_mask))
        x = self.sublayer[1](x, lambda x: self.src_attn(x, m, m, src_mask))
        return self.sublayer[2](x, self.feed_forward)</code></pre>
</div>
</div>


---

## Multi-head Attention

<div style="display: flex; align-items: flex-start; gap: 20px;">
<!-- Left: Diagram -->
<div style="flex: 1; text-align: center; max-width: 50%;">
  <img src="https://cdn.markslides.ai/users/1657/images/m-DOW7Hg69yqZdPHoOFla" width="100%" alt="Encoder Layer Diagram">
  <p style="font-size: 20px; margin-top: 10px;">
    [batch_size, sequence_length, d_model] <br>
    -> [batch_size, sequence_length, h, d_k] <br>
    -> [batch_size, h, sequence_length, d_k] <br>
    -> [batch_size, sequence_length, d_model]
  </p>
</div>

<!-- Right: Code Blocks with Paragraphs -->
<div style="flex: 1; max-width: 70%; display: flex; flex-direction: column; gap: 10px;">
  
  <!-- First code block -->
  <pre style="background-color: #f6f8fa; padding: 12px; border-radius: 5px; margin: 0 0 15px 0;"><code style="font-family: 'Consolas', monospace; font-size: 16px; line-height: 1.4;">class MultiHeadedAttention(nn.Module):
    def __init__(self, h, d_model, dropout=0.1):
        "Take in model size and number of heads."   
        self.d_k = d_model // h
        self.h = h
        self.linears = clones(nn.Linear(d_model, d_model), 4)

    def forward(self, query, key, value, mask=None):
        # 1) Do all the linear projections in batch from d_model => h x d_k
        query, key, value = [
          lin(x).view(nbatches, -1, self.h, self.d_k).transpose(1, 2)
            for lin, x in zip(self.linears, (query, key, value))]
            
        # 2) Apply attention on all the projected vectors in batch.
        x, self.attn = attention(query, key, value, mask=mask, dropout=self.dropout)
        
        # 3) "Concat" using a view and apply a final linear.
        x = x.transpose(1, 2).contiguous().
        view(nbatches, -1, self.h * self.d_k)
        return self.linears[-1](x)</code></pre>
</div>
</div>


---

## Feedforward layers
<!-- 
input (d_model) → Linear(d_ff) → ReLU → Dropout → Linear(d_model)

<div style="display: flex; align-items: center; gap: 20px;">

  <!-- Left: Centered Image -->
  <!-- <img src="https://cdn.markslides.ai/users/1657/images/EhXXnJuJ9BhAiAdCh7h4a" alt="Feedforward Diagram" style="width: 250px; display: block;" />

  <!-- Right: Code Block -->
  <!-- <pre><code class="language-python">
class PositionwiseFeedForward(nn.Module):
    def __init__(self, d_model, d_ff, dropout=0.1):
        super(PositionwiseFeedForward, self).__init__()
        self.w_1 = nn.Linear(d_model, d_ff)
        self.w_2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        return self.w_2(self.dropout(torch.relu(self.w_1(x))))
  </code></pre>

</div> --> --> -->

<div style="display: flex; align-items: flex-start; gap: 20px;">
<!-- Left: Diagram -->
<div style="flex: 1; text-align: center; max-width: 50%;">
  <img src="https://cdn.markslides.ai/users/1657/images/EhXXnJuJ9BhAiAdCh7h4a" width="100%" alt="Encoder Layer Diagram">
  <p style="font-size: 20px; margin-top: 10px;">
    input x: [batch_size, sequence_length, d_model] <br>
    -> [batch_size, sequence_length, 4 * d_model] <br>
    output: [batch_size, sequence_length, d_model]
  </p>
</div>

<!-- Right: Code Blocks with Paragraphs -->
<div style="flex: 1; max-width: 50%; display: flex; flex-direction: column; gap: 10px;">

  
  <!-- Second code block -->
  <pre style="background-color: #f6f8fa; padding: 12px; border-radius: 5px; margin: 0;"><code style="font-family: 'Consolas', monospace; font-size: 16px; line-height: 1.4;">class PositionwiseFeedForward(nn.Module):
    def __init__(self, d_model, d_ff, dropout=0.1):
        super(PositionwiseFeedForward, self).__init__()
        self.w_1 = nn.Linear(d_model, d_ff)
        self.w_2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        return self.w_2(self.dropout(torch.relu(self.w_1(x))))</code></pre>
</div>
</div>


---

# Training
---


## Training Overview

- Loss: CrossEntropy with label smoothing (0.1)
- Optimizer: Adam with Noam-style learning rate scheduler
- (Our implementation used 3000 warumup steps)



---
## Learning Rate Scheduling
![](https://cdn.markslides.ai/users/1557/images/JPFj7bWzM9wzvQvH7vZJx)

---
## Learning Rate Scheduling
- Warmup steps = 4,000
- Total steps = 100,000
![bg right:67% h:540](https://cdn.markslides.ai/users/1557/images/M3GKyHpbS_QYTTmXjXc2v)

---

## Label Smoothing


- Prevents overconfidence in model predictions
- Helps regularize the model
- Label smoothing = 0.1 used in our implementation

  | Label      |  class 1   |  class 2   | class 3    |
  |------------|------------|------------|------------|
  | Hard Label | 0          | 0          | 1          |
  | Smoothed Label | 0.05    | 0.05      | 0.9        |

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
## Potential issues of our implementation
- Our warmup (3000) was likely too large for the total training steps
  - Config 1 (BS=128): ~2200 accum steps after 100 epochs (136%)
  - Config 2 (BS=32): ~8900 accum steps after 50 epochs (34%)
- Original Paper:
  - 4000 warmup steps for total of 100,000 steps
![bg right width:640](https://cdn.markslides.ai/users/1557/images/R_hPfuVtdzqiDitsAEBwp)
<!-- ![bg right:67% width:960](https://cdn.markslides.ai/users/1557/images/olw9-wL-QF2a3tGi_8LeR)
 -->

---
# Distributed Training
---

## Distributed Training Overview

- PyTorch Distributed Data Parallel (DDP) framework
  - `torch.nn.parallel.DistributedDataParallel` (Model)
  - `torch.utils.data.DistributedSampler` (Dataset)
- Parallelized across 4 NVIDIA A100 GPUs on a single node
- Full model parameters kept on each GPU
- Gradient synchronization at backward pass

---

## Distributed Training Implementation Overview

- Implementation steps:
  - Initialize process groups
  - Move data to GPUs (model/optimizer/datasets/etc..)
  - Reduce gradients during backward pass
  - Broadcast updated weights
  - Reduce values for logging

---
## Computational Platform

- We used single GPU node on Perlmutter supercomputer at NERSC
- Single GPU node configuration:
  - CPU: 1 x AMD EPYC 7763 (Milan)
  - GPU: 4 x NVIDIA A100 (Ampere) GPUs
    - 40GB HBM2 with 1555 GB/s memory bandwidth
    - 3rd generation NVLink connections:
      - 4 links between each GPU pair
      - 25 GB/s/direction per link
---
# Evaluation

---
## Understanding Train and Validation Loss

- Training Loss: Measures how well the model fits the training data. Lower means better learning.
- Validation Loss: Measures model performance on unseen validation data.
- Important to monitor both to detect:
  - Underfitting (both high)
  - Overfitting (train low, val high)
- Compares predicted word probabilities vs true words
- Lower Cross-Entropy = better matching of outputs to targets
- We track Training Loss (on train data) and Validation Loss (on unseen data) across epochs.


---
## Training/Validation Loss
![bg right:60% w:720](https://cdn.markslides.ai/users/1557/images/_k3d43Qt3KFBpHZudRvz0)
- Training loss continuously decreases
- Validation loss exhibit plateau (then even increases)
- Gap between train/val loss indicates overfitting
---
## Training Results (Config 1)

- Batch size 128, Accumulation steps 21
- Tokens/accum : approx. 25,000
- 100 epochs completed
- Total Accumulation steps 2200

---

## Evaluation: Training Results (Config 2)

- Batch size 32, Accumulation steps 10
- Tokens/accum : approx. 4,000
- 54 epochs completed
  - Stopped early because it took very long and it was obvious that it was overfitting
- Total Accumulation steps 9612

---
## BLEU Score
- BLEU (Bilingual Evaluation Understudy):
  - Measures similarity between machine translation and human translation
  - Based on overlapping n-grams (word groups)
  - Score range: 0 (worst) to 100 (perfect)
- Why not simple accuracy?
  - Translation allows multiple correct outputs
  - Accuracy expects exact match (too strict)
- BLEU is industry standard for translation quality

---
## BLEU Score Evaluation

- Loaded trained model weights directly (.pt checkpoints)
- Compared outputs against multiple human references
- Evaluation process:
  - Decode each input sentence
  - Tokenize output and references
  - Calculate BLEU using SacreBLEU library
  - tqdm for batching and progress.
- Evaluated on 55–100 examples for faster experimentation


---
## BLEU Results and Observations
- Model 1 (batch size 32): BLEU = 0.31
- Model 2 (batch size 128): BLEU = TBD (after running)
- Observations:
  - BLEU scores are lower due to simple decoding strategy
  - Model 2 expected to perform better with larger batch and smoother loss curve
  - BLEU variation between models reflects real differences in training quality
- Improvements:
  - Optimize batch size, learning rate, accumulation.
  - Increase BLEU evaluation samples.
  - Try beam search decoding.
  - Fine-tune on larger datasets.

---

## Demo and Attention Visualization

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

## Overfitting Analysis : Sentence Length (number of tokens in sentence)
| Dataset     | Avg.     | Med.     | std.     |
|-------------|----------|----------|----------|
| Tatoeba zh  | 7.55    | 7.0    |3.98     |
| Tatoeba en  | 8.22    |7.0     | 4.20     |
| WMT14 de    | 28.12    | 24.00   |20.77   |
| WMT14 en    | 28.09    | 24.00    | 24.50    |

![bg right vertical h:280](https://cdn.markslides.ai/users/1657/images/iB5VUlz1aAQJxVy0tvpsp)
![bg right h:280](https://cdn.markslides.ai/users/1657/images/tU-5wRaQculcWIryIUYgq)

---
## Training Performance Study (on 4× A100 GPUs)
- Larger batch size -> Better performance
- Optimal performance at batch size 512 (total execution time)
- Maximum throughput at batch size 1024 (tokens/sec)
- Diminishing returns observed after batch size 512

![bg right h:720](https://cdn.markslides.ai/users/1557/images/p6HRs3LWMlBV0aKN07iHs)
---

## Training Performance Study (on 4× A100 GPUs)

| Batch Size | Accum. Steps | Exec. Time | Training Time | Tokens/Sec |
|------------|--------------|------------|---------------|------------|
| 16         | 64           | 314.00s    | 253.84s       | ~2,090     |
| 32         | 32           | 184.91s    | 130.85s       | ~4,040     |
| 64         | 16           | 118.89s    | 68.95s        | ~7,760     |
| 128        | 8            | 95.20s     | 47.22s        | ~11,360    |
| 256        | 4            | 89.86s     | 38.96s        | ~13,900    |
| 512        | 2            | 86.43s     | 35.54s        | ~15,250    |
| 1024       | 1            | 102.82s    | 33.33s        | ~16,150    |


---

## Conclusion

- Successfully implemented full Transformer architecture for Chinese-English MT
- Demonstrated learning dynamics across different batch sizes
- Observed common seq2seq training patterns: fast initial learning, eventual overfitting
- Gained practical experience with distributed training on HPC


---

## Additional Directions

- Improve tokenization using subword techniques (BPE, WordPiece)
- Test on more diverse datasets beyond Tatoeba
- Explore multilingual models (zh → multiple languages)


---

## References

- Vaswani et al., "Attention is All You Need", NeurIPS 2017
- The Annotated Transformer (Harvard NLP)
- PyTorch Distributed: https://pytorch.org/tutorials/intermediate/dist_tuto.html
- Tatoeba Project: https://tatoeba.org/
- NERSC Perlmutter: https://docs.nersc.gov/systems/perlmutter/architecture/


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
