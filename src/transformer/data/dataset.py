import torch
import torchtext
from torch.utils.data import IterableDataset
import spacy
from typing import List, Tuple, Dict, Callable, Iterator, Optional
from torch.utils.data import random_split
from collections import Counter, OrderedDict


class TSVTranslationDataset(IterableDataset):
    """Dataset for loading translation pairs from TSV files."""

    def __init__(self, path: str, src_col: int = 1, tgt_col: int = 3, sep: str = "\t"):
        self.path = path
        self.src_col = src_col
        self.tgt_col = tgt_col
        self.sep = sep

    def __iter__(self):
        with open(self.path, "r", encoding="utf-8") as f:
            for line in f:
                parts = line.strip().split(self.sep)
                if len(parts) > max(self.src_col, self.tgt_col):
                    yield parts[self.src_col], parts[self.tgt_col]


def load_tokenizers(model_names: Dict[str, str]) -> Dict[str, spacy.language.Language]:
    """Load spacy tokenizers for different languages."""
    result = {}
    for lang, model in model_names.items():
        # Check if model is already downloaded
        try:
            # Try to load the model directly first
            nlp = spacy.load(model)
            result[lang] = nlp
        except OSError:
            # If model is not found, download it
            print(f"Model {model} not found. Downloading...")
            spacy.cli.download(model)
            # Load the newly downloaded model
            result[lang] = spacy.load(model)
    return result


def tokenize(text: str, nlp: spacy.language.Language) -> List[str]:
    """Tokenize text using spacy tokenizer."""
    return [tok.text for tok in nlp.tokenizer(text)]


class Vocab:
    """Custom vocabulary class to replace torchtext.vocab.Vocab."""
    
    def __init__(self, counter: Counter, min_freq: int = 1, specials: Optional[List[str]] = None):
        """Initialize vocabulary from counter and special tokens.
        
        Args:
            counter: Counter object containing token frequencies
            min_freq: Minimum frequency for a token to be included
            specials: List of special tokens to add to vocabulary
        """
        self.specials = specials if specials is not None else []
        self.min_freq = min_freq
        self.default_index = 0  # Default to first token (usually <unk>)
        
        # Filter tokens by minimum frequency
        filtered_tokens = [(token, freq) for token, freq in counter.items() 
                          if freq >= min_freq]
        
        # Sort by frequency (descending) and then alphabetically
        sorted_tokens = sorted(filtered_tokens, key=lambda x: (-x[1], x[0]))
        
        # Create ordered dictionary with special tokens first, then sorted tokens
        self.stoi = OrderedDict()
        self.itos = []
        
        # Add special tokens
        for token in self.specials:
            self.stoi[token] = len(self.stoi)
            self.itos.append(token)
            
        # Add regular tokens
        for token, _ in sorted_tokens:
            if token not in self.stoi:
                self.stoi[token] = len(self.stoi)
                self.itos.append(token)
    
    def __len__(self) -> int:
        return len(self.stoi)
    
    def __getitem__(self, token: str) -> int:
        """Get index for a token."""
        return self.stoi.get(token, self.default_index)
    
    def __call__(self, tokens: List[str]) -> List[int]:
        """Convert list of tokens to indices."""
        return [self[token] for token in tokens]
    
    def get_stoi(self) -> Dict[str, int]:
        """Get string-to-index mapping."""
        return dict(self.stoi)
    
    def get_itos(self) -> List[str]:
        """Get index-to-string mapping."""
        return self.itos.copy()
        
    def set_default_index(self, index: int) -> None:
        """Set the default index to return for unknown tokens."""
        self.default_index = index


def build_vocab(
    iterator: Iterator[str], tokenize_fn: Callable, min_freq: int, specials: List[str]
) -> Vocab:
    """Build vocabulary from iterator."""
    
    counter = Counter()
    for text in iterator:
        counter.update(tokenize_fn(text))
    
    return Vocab(counter, min_freq=min_freq, specials=specials)


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
        tgt_mask = tgt_mask & subsequent_mask(tgt.size(-1)).type_as(tgt_mask.data)
        return tgt_mask


def make_collate_fn(config, tokenizers, vocab_src, vocab_tgt, device, max_len=128):
    """Create collate function for DataLoader."""

    def collate(batch):
        src_bos_id = vocab_src["<s>"]
        src_eos_id = vocab_src["</s>"]
        src_pad_id = vocab_src["<blank>"]
        tgt_bos_id = vocab_src["<s>"]
        tgt_eos_id = vocab_src["</s>"]
        tgt_pad_id = vocab_tgt["<blank>"]
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
            src_indices = torch.tensor(
                vocab_src(src_tokens)[: max_len - 2], dtype=torch.int64, device=device
            )
            tgt_indices = torch.tensor(
                vocab_tgt(tgt_tokens)[: max_len - 2], dtype=torch.int64, device=device
            )

            src_pad_len = max(0, max_len - len(src_indices) - 2)
            tgt_pad_len = max(0, max_len - len(tgt_indices) - 2)

            src_batch.append(
                torch.cat(
                    tensors=[
                        src_bos_id,
                        src_indices,
                        src_eos_id,
                        torch.full((src_pad_len,), src_pad_id, device=device),
                    ],
                    dim=0,
                )
            )
            tgt_batch.append(
                torch.cat(
                    tensors=[
                        tgt_bos_id,
                        tgt_indices,
                        tgt_eos_id,
                        torch.full((tgt_pad_len,), tgt_pad_id, device=device),
                    ],
                    dim=0,
                )
            )

        src = torch.stack(src_batch)
        tgt = torch.stack(tgt_batch)

        # Convert to tensors
        return Batch(src, tgt, src_pad_id)

    return collate


def create_dataloaders(
    config, tokenizers, vocab_src, vocab_tgt, device, max_len=128, distributed=False
):
    """Create train, validation and test dataloaders."""

    def wrap(ds):
        return torch.utils.data.DataLoader(
            ds,
            batch_size=config.batch_size,
            collate_fn=make_collate_fn(
                config, tokenizers, vocab_src, vocab_tgt, device, max_len
            ),
            shuffle=not distributed,
        )

    train_ds, val_ds, test_ds = config.dataset_loader()

    if distributed:
        train_sampler = torch.utils.data.distributed.DistributedSampler(train_ds)
        train_dl = torch.utils.data.DataLoader(
            train_ds,
            batch_size=config.batch_size,
            collate_fn=make_collate_fn(
                config, tokenizers, vocab_src, vocab_tgt, device, max_len
            ),
            sampler=train_sampler,
            num_workers=4,
            pin_memory=True,
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
    torchtext.datasets.multi30k.URL["train"] = (
        "https://raw.githubusercontent.com/neychev/small_DL_repo/master/datasets/Multi30k/training.tar.gz"
    )
    torchtext.datasets.multi30k.URL["valid"] = (
        "https://raw.githubusercontent.com/neychev/small_DL_repo/master/datasets/Multi30k/validation.tar.gz"
    )
    torchtext.datasets.multi30k.MD5["train"] = (
        "20140d013d05dd9a72dfde46478663ba05737ce983f478f960c1123c6671be5e"
    )
    torchtext.datasets.multi30k.MD5["valid"] = (
        "a7aa20e9ebd5ba5adce7909498b94410996040857154dab029851af3a866da8c"
    )

    train = torchtext.datasets.Multi30k(root=".data", split="train", language_pair=("de", "en"))
    val = torchtext.datasets.Multi30k(root=".data", split="valid", language_pair=("de", "en"))
    # TODO: This is not ideal, test dataset should not be identical to validation dataset
    test = torchtext.datasets.Multi30k(root=".data", split="valid", language_pair=("de", "en"))
    return train, val, test


def tatoeba_zh_en_loader(
    path: str = ".data/datasets/tatoeba/zh_en/Sentence pairs in Mandarin Chinese-English - 2025-04-27.tsv",
    src_col: int = 1,
    tgt_col: int = 3,
    sep: str = "\t",
    split_ratio: Tuple[float, float, float] = (0.8, 0.1, 0.1),
    seed: int = 42,
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
        generator=torch.Generator().manual_seed(seed),
    )

    return train_data, val_data, test_data
