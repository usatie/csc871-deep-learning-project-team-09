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
    """Implement label smoothing."""

    def __init__(self, size: int, padding_idx: int, smoothing: float = 0.0):
        super(LabelSmoothing, self).__init__()
        self.criterion = nn.KLDivLoss(reduction="sum")
        self.padding_idx = padding_idx
        self.confidence = 1.0 - smoothing
        self.smoothing = smoothing
        self.size = size
        self.true_dist = None

    def forward(self, x: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        assert x.size(1) == self.size
        true_dist = x.data.clone()
        true_dist.fill_(self.smoothing / (self.size - 2))
        true_dist.scatter_(1, target.data.unsqueeze(1), self.confidence)
        true_dist[:, self.padding_idx] = 0
        mask = torch.nonzero(target.data == self.padding_idx)
        if mask.dim() > 0:
            true_dist.index_fill_(0, mask.squeeze(), 0.0)
        self.true_dist = true_dist
        return self.criterion(x, true_dist.clone().detach())


class SimpleLossCompute:
    """A simple loss compute and train function."""

    def __init__(self, generator: nn.Module, criterion: nn.Module):
        self.generator = generator
        self.criterion = criterion

    def __call__(self, x: torch.Tensor, y: torch.Tensor, norm: float) -> torch.Tensor:
        x = self.generator(x)
        sloss = (
            self.criterion(x.contiguous().view(-1, x.size(-1)), y.contiguous().view(-1))
            / norm
        )
        return sloss.data * norm, sloss


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

        loss, loss_node = loss_compute(out, batch.tgt_y, batch.ntokens)

        total_loss += loss
        total_tokens += batch.ntokens
        tokens += batch.ntokens

        if mode == "train" or mode == "train+log":
            loss_node.backward()
            train_state.step += 1
            train_state.samples += batch.src.size(0)
            train_state.tokens += batch.ntokens

            if i % accum_iter == 0:
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
                n_accum += 1
                train_state.accum_step += 1
            scheduler.step()

        if i % 40 == 1 and (mode == "train" or mode == "train+log"):
            lr = optimizer.param_groups[0]["lr"]
            elapsed = time.time() - start
            if distributed:
                dist.reduce(tokens, dst=0)
                dist.reduce(loss, dst=0)
                dist.reduce(batch.ntokens, dst=0)
            if rank == 0:
                print(
                    (
                        "Epoch Step: %6d | Accumulation Step: %3d | Loss: %6.2f "
                        + "| Tokens / Sec: %7.1f | Tokens: %6d | Learning Rate: %6.1e"
                    )
                    % (
                        i,
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


def greedy_decode(
    model: nn.Module,
    src: torch.Tensor,
    src_mask: torch.Tensor,
    max_len: int,
    start_symbol: int,
) -> torch.Tensor:
    """Greedy decoding for inference."""
    memory = model.encode(src, src_mask)
    batch_size = src.size(0)
    ys = torch.zeros(batch_size, 1).fill_(start_symbol).type_as(src.data)
    for i in range(max_len - 1):
        out = model.decode(
            memory, src_mask, ys, subsequent_mask(ys.size(1)).type_as(src.data)
        )
        prob = model.generator(out[:, -1])
        _, next_word = torch.max(prob, dim=1)
        next_word = next_word.data[0]
        ys = torch.cat(
            [ys, torch.zeros(batch_size, 1).type_as(src.data).fill_(next_word)], dim=1
        )
    return ys
