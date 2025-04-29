import torch
import torchtext
from torch.utils.data import Dataset, IterableDataset
import spacy
from typing import List, Tuple, Dict, Callable
import pandas as pd
import torchtext.datasets as datasets
from torch.utils.data import random_split

class TSVTranslationDataset(IterableDataset):
    """Dataset for loading translation pairs from TSV files."""
    def __init__(self, path: str, src_col: int = 1, tgt_col: int = 3, sep: str = "\t"):
        self.path = path
        self.src_col = src_col
        self.tgt_col = tgt_col
        self.sep = sep

    def __iter__(self):
        with open(self.path, 'r', encoding='utf-8') as f:
            for line in f:
                parts = line.strip().split(self.sep)
                if len(parts) > max(self.src_col, self.tgt_col):
                    yield parts[self.src_col], parts[self.tgt_col]

def load_tokenizers(model_names: Dict[str, str]) -> Dict[str, spacy.language.Language]:
    """Load spacy tokenizers for different languages."""
    return {lang: spacy.load(model) for lang, model in model_names.items()}

def tokenize(text: str, nlp: spacy.language.Language) -> List[str]:
    """Tokenize text using spacy tokenizer."""
    return [tok.text for tok in nlp.tokenizer(text)]

def build_vocab(iterator, tokenize_fn: Callable, min_freq: int, specials: List[str]) -> torch.nn.Module:
    """Build vocabulary from iterator."""
    def yield_tokens():
        for text in iterator:
            yield tokenize_fn(text)
    
    vocab = torchtext.vocab.build_vocab_from_iterator(
        yield_tokens(),
        min_freq=min_freq,
        specials=specials
    )
    return vocab

from transformer.model import subsequent_mask
class Batch:
    """Object for holding a batch of data with mask during training."""

    def __init__(self, src, tgt=None, pad=2):  # 2 = <blank>
        self.src = src
        self.src_mask = (src != pad).unsqueeze(-2)
        if tgt is not None:
            self.tgt = tgt[:, :-1]
            self.tgt_y = tgt[:, 1:]
            self.tgt_mask = self.make_std_mask(self.tgt, pad)
            self.ntokens = (self.tgt_y != pad).data.sum()

    @staticmethod
    def make_std_mask(tgt, pad):
        "Create a mask to hide padding and future words."
        tgt_mask = (tgt != pad).unsqueeze(-2)
        tgt_mask = tgt_mask & subsequent_mask(tgt.size(-1)).type_as(
            tgt_mask.data
        )
        return tgt_mask

def make_collate_fn(config, tokenizers, vocab_src, vocab_tgt, device, max_len=128):
    """Create collate function for DataLoader."""
    def collate(batch):
        src_bos_id = vocab_src['<s>']
        src_eos_id = vocab_src['</s>']
        src_pad_id = vocab_src['<blank>']
        tgt_bos_id = vocab_src['<s>']
        tgt_eos_id = vocab_src['</s>']
        tgt_pad_id = vocab_tgt['<blank>']
        assert src_bos_id == tgt_bos_id
        assert src_eos_id == tgt_eos_id
        assert src_pad_id == tgt_pad_id
        src_bos_id = torch.tensor([src_bos_id], device=device)  # <s> token id
        src_eos_id = torch.tensor([src_eos_id], device=device)  # </s> token id
        tgt_bos_id = torch.tensor([tgt_bos_id], device=device)  # <s> token id
        tgt_eos_id = torch.tensor([tgt_eos_id], device=device)  # </s> token id

        src_batch, tgt_batch = [], []
        for src_sample, tgt_sample in batch:
            # Convert to string if not already
            src_text = str(src_sample)
            tgt_text = str(tgt_sample)
            
            # Tokenize
            src_tokens = tokenize(src_text, tokenizers[config.src_lang])
            tgt_tokens = tokenize(tgt_text, tokenizers[config.tgt_lang])
            
            # Convert tokens to indices
            src_indices = torch.tensor(vocab_src(src_tokens)[:max_len-2], dtype=torch.int64, device=device)
            tgt_indices = torch.tensor(vocab_tgt(tgt_tokens)[:max_len-2], dtype=torch.int64, device=device)
            
            src_pad_len = max(0, max_len-len(src_indices)-2)
            tgt_pad_len = max(0, max_len-len(tgt_indices)-2)
            
            src_batch.append(torch.cat(
                tensors=[
                    src_bos_id,
                    src_indices,
                    src_eos_id,
                    torch.full((src_pad_len,), src_pad_id, device=device)
                ],
                dim=0
            ))
            tgt_batch.append(torch.cat(
                tensors=[
                    tgt_bos_id,
                    tgt_indices,
                    tgt_eos_id,
                    torch.full((tgt_pad_len,), tgt_pad_id, device=device)
                ],
                dim=0
            ))
            
        src = torch.stack(src_batch)
        tgt = torch.stack(tgt_batch)

        # Convert to tensors
        return Batch(src, tgt, src_pad_id)
    
    return collate

def create_dataloaders(
    config,
    tokenizers,
    vocab_src,
    vocab_tgt,
    device,
    batch_size=12000,
    max_len=128,
    distributed=False
):
    """Create train, validation and test dataloaders."""
    def wrap(ds):
        return torch.utils.data.DataLoader(
            ds,
            batch_size=batch_size,
            collate_fn=make_collate_fn(config, tokenizers, vocab_src, vocab_tgt, device, max_len),
            shuffle=not distributed,
        )
    
    train_ds, val_ds, test_ds = config.dataset_loader()
    
    if distributed:
        train_sampler = torch.utils.data.distributed.DistributedSampler(train_ds)
        train_dl = torch.utils.data.DataLoader(
            train_ds,
            batch_size=batch_size,
            collate_fn=make_collate_fn(config, tokenizers, vocab_src, vocab_tgt, device, max_len),
            sampler=train_sampler,
            num_workers=4,
            pin_memory=True
        )
    else:
        train_dl = wrap(train_ds)
    
    val_dl = wrap(val_ds)
    test_dl = wrap(test_ds)
    
    return train_dl, val_dl, test_dl

def multi30k_loader():
    """
    The code below is a workaround for this code

    train, val, test = datasets.Multi30k(language_pair=("de", "en"))
    return train, val, test
    """
    from torchtext.datasets import multi30k
    multi30k.URL["train"] = "https://raw.githubusercontent.com/neychev/small_DL_repo/master/datasets/Multi30k/training.tar.gz"
    multi30k.URL["valid"] = "https://raw.githubusercontent.com/neychev/small_DL_repo/master/datasets/Multi30k/validation.tar.gz"
    multi30k.MD5["train"] = "20140d013d05dd9a72dfde46478663ba05737ce983f478f960c1123c6671be5e"
    multi30k.MD5["valid"] = "a7aa20e9ebd5ba5adce7909498b94410996040857154dab029851af3a866da8c"

    train = datasets.Multi30k(root='.data', split='train', language_pair=('de', 'en'))
    val = datasets.Multi30k(root='.data', split='valid', language_pair=('de', 'en'))
    # TODO: This is not ideal, test dataset should not be identical to validation dataset
    test = datasets.Multi30k(root='.data', split='valid', language_pair=('de', 'en'))
    return train, val, test

def tatoeba_zh_en_loader(
    path: str = "Sentence pairs in Mandarin Chinese-English - 2025-04-27.tsv",
    src_col: int = 1,
    tgt_col: int = 3,
    sep: str = "\t",
    split_ratio: Tuple[float, float, float] = (0.8, 0.1, 0.1),
    seed: int = 42
):
    """Load Tatoeba Chinese-English dataset from TSV file."""
    # Create dataset
    dataset = TSVTranslationDataset(path, src_col, tgt_col, sep)
    
    # Convert to list for splitting
    data = list(dataset)
    
    # Calculate split sizes
    total_size = len(data)
    train_size = int(total_size * split_ratio[0])
    val_size = int(total_size * split_ratio[1])
    test_size = total_size - train_size - val_size
    
    # Split dataset
    train_data, val_data, test_data = random_split(
        data,
        [train_size, val_size, test_size],
        generator=torch.Generator().manual_seed(seed)
    )
    
    return train_data, val_data, test_data 