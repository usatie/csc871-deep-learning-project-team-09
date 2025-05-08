import time
from typing import Optional

import torch
import torch.nn as nn
from torch.optim.lr_scheduler import LambdaLR
import torch.distributed as dist

from transformer.model import subsequent_mask


class TrainState:
    """Track number of steps, examples, and tokens processed"""

    step: int = 0  # Steps in the current epoch
    accum_step: int = 0  # Number of gradient accumulation steps
    samples: int = 0  # total # of examples used
    tokens: int = 0  # total # of tokens processed


class LabelSmoothing(nn.Module):
    """
    Implements label smoothing using KL divergence loss.

    Note on tokens:
    - The 'size' parameter represents the vocabulary size (number of possible tokens)
    - When calculating loss, we operate on flattened token predictions and labels
    - Padding tokens are explicitly excluded from loss calculation to avoid skewing gradients
    - Each non-padding token contributes equally to the loss regardless of sequence length

    Returns:
        The KL divergence loss, which is not mean value of the loss across all tokens,
        but the sum of the loss across all tokens. This allows for explicit normalization in SimpleLossCompute.
    """

    def __init__(self, size: int, padding_idx: int, smoothing: float = 0.0):
        super(LabelSmoothing, self).__init__()
        # Using "sum" reduction to get total loss across all tokens
        self.criterion = nn.KLDivLoss(reduction="sum")
        self.padding_idx = padding_idx
        self.confidence = 1.0 - smoothing
        self.smoothing = smoothing
        self.size = size
        self.true_dist = None

    def forward(self, x: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        # x.size(1) is the vocabulary dimension containing predictions for each token
        assert x.size(1) == self.size
        true_dist = x.data.clone()
        # Distribute smoothing probability mass across non-special tokens
        # Note: We subtract 2 to account for padding token and the correct token
        true_dist.fill_(self.smoothing / (self.size - 2))
        # Assign high probability (confidence) to correct tokens
        true_dist.scatter_(1, target.data.unsqueeze(1), self.confidence)
        # Zero out padding token predictions to avoid counting them in loss
        true_dist[:, self.padding_idx] = 0
        mask = torch.nonzero(target.data == self.padding_idx)
        if mask.dim() > 0:
            # Zero out all predictions for padding positions
            true_dist.index_fill_(0, mask.squeeze(), 0.0)
        self.true_dist = true_dist
        return self.criterion(x, true_dist.clone().detach())


class SimpleLossCompute:
    """
    Handles loss computation with proper token-based normalization.

    Critical token normalization aspects:
    - The 'norm' parameter represents the total number of non-padding tokens in the batch
    - For training purposes, loss is normalized by dividing by token count to get per-token loss
    - For logging purposes, we also return the un-normalized loss
    - This token normalization ensures stable gradients regardless of varying sequence lengths
    - Without this normalization, longer sequences would dominate training signal
    """

    def __init__(self, generator: nn.Module, criterion: nn.Module):
        self.generator = generator
        self.criterion = criterion

    def __call__(
        self, x: torch.Tensor, y: torch.Tensor, norm: float
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        x = self.generator(x)
        # Reshape tensors to [batch_size*seq_len, vocab_size] and [batch_size*seq_len]
        # This flattening treats each token prediction independently
        loss = self.criterion(
            x.contiguous().view(-1, x.size(-1)), y.contiguous().view(-1)
        )

        # Normalize loss by dividing by the number of tokens (excluding padding)
        # This ensures that each token contributes equally to the gradient
        normalized_loss = loss / norm

        # Return both the un-normalized loss (for logging) and normalized loss (for backprop)
        # The un-normalized loss (loss.data * norm) helps track total loss magnitude
        # The normalized loss (loss/norm) ensures balanced gradients for stable training
        return loss.data, normalized_loss


def rate(step: int, model_size: int, factor: float, warmup: int) -> float:
    """
    we have to default the step to 1 for LambdaLR function
    to avoid zero raising to negative power.
    """
    if step == 0:
        step = 1
    return factor * (
        model_size ** (-0.5) * min(step ** (-0.5), step * warmup ** (-1.5))
    )


def run_epoch(
    data_iter,
    model: nn.Module,
    loss_compute: SimpleLossCompute,
    optimizer: torch.optim.Optimizer,
    scheduler: Optional[LambdaLR],
    rank: int,
    distributed: bool,
    mode: str = "train",
    accum_iter: int = 1,
    train_state: TrainState = TrainState(),
) -> float:
    """Train a single epoch"""
    start = time.time()
    total_tokens = 0
    total_loss = 0
    tokens = 0
    n_accum = 0

    for i, batch in enumerate(data_iter):
        out = model.forward(batch.src, batch.tgt, batch.src_mask, batch.tgt_mask)

        # `out` does not (necessarily) have the `bos` token at first position
        # `batch.tgt_y` does not have the `bos` token at first position (removed in collate function)
        # `loss` is for recording purposes
        # `loss_node` is for training purposes
        loss, loss_node = loss_compute(out, batch.tgt_y, batch.ntokens)

        total_loss += loss
        total_tokens += batch.ntokens
        tokens += batch.ntokens

        if mode == "train" or mode == "train+log":
            loss_node.backward()
            train_state.step += 1
            train_state.samples += batch.src.size(0)
            train_state.tokens += batch.ntokens

            if (i + 1) % accum_iter == 0:
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
                n_accum += 1
                train_state.accum_step += 1
                scheduler.step()
                if distributed:
                    dist.reduce(tokens, dst=0)
                    dist.reduce(loss, dst=0)
                    dist.reduce(batch.ntokens, dst=0)
                if rank == 0:
                    lr = optimizer.param_groups[0]["lr"]
                    elapsed = time.time() - start
                    print(
                        (
                            "Epoch Step: %6d | Accumulation Step: %3d | Loss: %6.2f "
                            + "| Tokens / Sec: %7.1f | Tokens: %6d | Learning Rate: %6.1e"
                        )
                        % (
                            i + 1,
                            n_accum,
                            loss / batch.ntokens,
                            tokens / elapsed,
                            tokens,
                            lr,
                        )
                    )
                start = time.time()
                tokens = 0

        del loss
        del loss_node

    return total_loss / total_tokens, train_state
