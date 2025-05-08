import time
from typing import Optional, Tuple

import torch
import torch.nn as nn
from torch.optim.lr_scheduler import LambdaLR
import torch.distributed as dist


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
    lr = factor * (model_size ** (-0.5) * min(step ** (-0.5), step * warmup ** (-1.5)))
    return lr


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
) -> Tuple[float, TrainState]:
    """
    Train or evaluate a single epoch

    Args:
        data_iter: Iterator over batches
        model: The neural network model
        loss_compute: Loss computation function
        optimizer: Optimizer for parameter updates
        scheduler: Learning rate scheduler
        rank: Process rank in distributed training
        distributed: Whether using distributed training
        mode: 'train', 'train+log', or 'eval'
        accum_iter: Number of batches for gradient accumulation
        train_state: Current training state

    Returns:
        Tuple of (normalized loss, updated train state)
    """
    start = time.time()
    epoch_tokens = 0
    epoch_loss = 0
    accum_tokens = 0
    accum_loss = 0
    pending_backward = 0  # Initialize pending backward counter
    for i, batch in enumerate(data_iter):
        out = model.forward(batch.src, batch.tgt, batch.src_mask, batch.tgt_mask)

        # Get both raw loss and normalized loss node for backprop
        raw_loss, loss_node = loss_compute(out, batch.tgt_y, batch.ntokens)

        # Track cumulative loss and tokens
        batch_loss = raw_loss.clone()
        batch_tokens = batch.ntokens.clone()
        # Accumulate totals for epoch statistics
        epoch_loss += batch_loss
        epoch_tokens += batch_tokens
        # Accumulate values for logging
        accum_loss += batch_loss
        accum_tokens += batch_tokens
        if mode == "train" or mode == "train+log":
            # Backward pass for gradient accumulation
            loss_node.backward()
            train_state.step += 1
            train_state.samples += batch.src.size(0)
            train_state.tokens += batch_tokens.item()
            pending_backward += 1  # Increment pending backward counter

            # Perform optimizer step after accumulating gradients
            if (i + 1) % accum_iter == 0 or (i == len(data_iter) - 1):
                # Execute optimization step for either:
                # 1. Regular accumulation interval OR
                # 2. Last batch in the iterator
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
                train_state.accum_step += 1
                scheduler.step()

                # For the logging step, synchronize across GPUs if distributed
                if distributed:
                    # Create tensors to hold reduced values
                    reduced_accum_loss = accum_loss.clone()
                    reduced_accum_tokens = accum_tokens.clone()
                    reduced_epoch_loss = epoch_loss.clone()
                    reduced_epoch_tokens = epoch_tokens.clone()

                    # Reduce values across all GPUs
                    dist.reduce(reduced_accum_loss, dst=0)
                    dist.reduce(reduced_accum_tokens, dst=0)
                    dist.reduce(reduced_epoch_loss, dst=0)
                    dist.reduce(reduced_epoch_tokens, dst=0)

                    # Only rank 0 will have the correct values after reduction
                    if rank == 0:
                        accum_loss = reduced_accum_loss
                        accum_tokens = reduced_accum_tokens
                        epoch_loss = reduced_epoch_loss
                        epoch_tokens = reduced_epoch_tokens

                # Log training progress (only on rank 0)
                if rank == 0:
                    lr = optimizer.param_groups[0]["lr"]
                    elapsed = time.time() - start

                    # Calculate metrics for logging
                    tokens_per_sec = accum_tokens / elapsed if elapsed > 0 else 0
                    # Normalized loss for the current accumulation window
                    window_loss = accum_loss / accum_tokens if accum_tokens > 0 else 0
                    # Running average loss for the entire epoch so far
                    running_avg_loss = (
                        epoch_loss / epoch_tokens if epoch_tokens > 0 else 0
                    )

                    # Indicate if this was a partial accumulation step (for the last batch)
                    is_partial = (i == len(data_iter) - 1) and (
                        pending_backward < accum_iter
                    )
                    partial_str = (
                        " (Partial: %d/%d)" % (pending_backward, accum_iter)
                        if is_partial
                        else ""
                    )

                    print(
                        (
                            "Epoch Step: %6d | Accum Step: %3d%s | "
                            + "Current Loss: %6.2f | Running Avg Loss: %6.2f | "
                            + "Tokens/Sec: %7.1f | Batch Tokens: %6d | Total Tokens: %6d | "
                            + "Learning Rate: %6.1e"
                        )
                        % (
                            i + 1,
                            train_state.accum_step,
                            partial_str,
                            window_loss,
                            running_avg_loss,
                            tokens_per_sec,
                            accum_tokens,
                            epoch_tokens,
                            lr,
                        ),
                        flush=True,
                    )

                # Reset counters for the next logging window
                start = time.time()
                accum_tokens = 0
                accum_loss = 0
                pending_backward = 0  # Reset pending backward counter
        # Clean up to avoid memory leaks
        del loss_node
        del raw_loss

    # Compute final loss for the epoch
    # For distributed training, make sure all processes have the same values
    if distributed:
        reduced_epoch_loss = epoch_loss.clone()
        reduced_epoch_tokens = epoch_tokens.clone()
        dist.reduce(reduced_epoch_loss, dst=0)
        dist.reduce(reduced_epoch_tokens, dst=0)

        # Broadcast the reduced values to all processes
        dist.broadcast(reduced_epoch_loss, src=0)
        dist.broadcast(reduced_epoch_tokens, src=0)

        epoch_loss = reduced_epoch_loss
        epoch_tokens = reduced_epoch_tokens

    # Compute normalized loss for the entire epoch
    normalized_loss = epoch_loss / epoch_tokens if epoch_tokens > 0 else 0

    return normalized_loss, train_state
