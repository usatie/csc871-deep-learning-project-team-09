import torch
import torch.nn as nn

from transformer.model import subsequent_mask


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
