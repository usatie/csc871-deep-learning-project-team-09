# Standard library imports
import os
import urllib.request
import tarfile
import hashlib
from typing import List, Tuple, Dict, Callable, Iterator, Optional
from collections import Counter, OrderedDict

# Third-party imports
import torch
import spacy
from torch.utils.data import random_split, IterableDataset

# Local imports
from transformer.model import subsequent_mask


# Helper functions
def to_map_style_dataset(iterable_dataset):
    r"""Convert iterable-style dataset to map-style dataset.

    args:
        iterable_dataset: An iterator type object. Examples include Iterable datasets, string list, text io, generators etc.
    """

    # Inner class to convert iterable-style to map-style dataset
    class _MapStyleDataset(torch.utils.data.Dataset):
        def __init__(self, iterable_dataset) -> None:
            # TODO Avoid list issue #1296
            self._data = list(iterable_dataset)

        def __len__(self):
            return len(self._data)

        def __getitem__(self, idx):
            return self._data[idx]

    return _MapStyleDataset(iterable_dataset)


def tokenize(text: str, tokenizer: spacy.language.Language) -> List[str]:
    """Tokenize text using spacy tokenizer."""
    return [token.text for token in tokenizer.tokenizer(text)]


def verify_sha256(file_path: str, expected_hash: str) -> bool:
    """Verify the SHA-256 hash of a file.

    Args:
        file_path: Path to the file to verify
        expected_hash: Expected SHA-256 hash in hexadecimal string format

    Returns:
        True if the hash matches, False otherwise
    """
    sha256_hash = hashlib.sha256()
    with open(file_path, "rb") as file:
        # Read the file in chunks to handle large files
        for chunk in iter(lambda: file.read(4096), b""):
            sha256_hash.update(chunk)
    return sha256_hash.hexdigest() == expected_hash


# Core classes
class Vocab:
    """Custom vocabulary class to replace torchtext.vocab.Vocab."""

    def __init__(
        self,
        token_counter: Counter,
        min_freq: int = 1,
        specials: Optional[List[str]] = None,
    ):
        """Initialize vocabulary from counter and special tokens.

        Args:
            token_counter: Counter object containing token frequencies
            min_freq: Minimum frequency for a token to be included
            specials: List of special tokens to add to vocabulary
        """
        self.specials = specials if specials is not None else []
        self.min_freq = min_freq
        self.default_index = 0  # Default to first token (usually <unk>)

        # Filter tokens by minimum frequency
        filtered_tokens = [
            (token, freq) for token, freq in token_counter.items() if freq >= min_freq
        ]

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


# Dataset classes
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


class Multi30kDataset(IterableDataset):
    """Dataset for loading Multi30k translation pairs."""

    def __init__(self, de_path: str, en_path: str):
        self.de_path = de_path
        self.en_path = en_path

    def __iter__(self):
        with open(self.de_path, "r", encoding="utf-8") as de_file, open(
            self.en_path, "r", encoding="utf-8"
        ) as en_file:
            for de_line, en_line in zip(de_file, en_file):
                yield de_line.strip(), en_line.strip()


# DataLoader and collation functions
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

    def wrap(dataset):
        # Don't shuffle for IterableDataset
        shuffle = not distributed
        map_style_dataset = to_map_style_dataset(dataset)
        return torch.utils.data.DataLoader(
            map_style_dataset,
            batch_size=config.batch_size,
            collate_fn=make_collate_fn(
                config, tokenizers, vocab_src, vocab_tgt, device, max_len
            ),
            shuffle=shuffle,
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


# Dataset loading utilities
def load_tokenizers(model_names: Dict[str, str]) -> Dict[str, spacy.language.Language]:
    """Load spacy tokenizers for different languages."""
    tokenizers = {}
    for lang, model in model_names.items():
        # Check if model is already downloaded
        try:
            # Try to load the model directly first
            nlp = spacy.load(model)
            tokenizers[lang] = nlp
        except OSError:
            # If model is not found, download it
            print(f"Model {model} not found. Downloading...")
            spacy.cli.download(model)
            # Load the newly downloaded model
            tokenizers[lang] = spacy.load(model)
    return tokenizers


def build_vocab(
    text_iterator: Iterator[str],
    tokenize_fn: Callable,
    min_freq: int,
    specials: List[str],
) -> Vocab:
    """Build vocabulary from iterator."""

    token_counter = Counter()
    for text in text_iterator:
        token_counter.update(tokenize_fn(text))

    return Vocab(token_counter, min_freq=min_freq, specials=specials)


def download_multi30k(root_dir: str = ".data/datasets/multi30k") -> Dict[str, str]:
    """Download Multi30k dataset files if they don't exist.

    Args:
        root_dir: Root directory to store the dataset

    Returns:
        Dictionary mapping split names to their file paths
    """
    base_url = "https://raw.githubusercontent.com/neychev/small_DL_repo/master/datasets/Multi30k"
    sha256_hashes = {
        "train": "20140d013d05dd9a72dfde46478663ba05737ce983f478f960c1123c6671be5e",
        "valid": "a7aa20e9ebd5ba5adce7909498b94410996040857154dab029851af3a866da8c",
    }
    urls = {
        "train": f"{base_url}/training.tar.gz",
        "valid": f"{base_url}/validation.tar.gz",
    }
    file_prefixes = {"train": "train", "valid": "val"}

    # Create root directory if it doesn't exist
    os.makedirs(root_dir, exist_ok=True)

    file_paths = {}
    for split in ["train", "valid"]:
        url = urls[split]
        tar_path = os.path.join(root_dir, f"{split}.tar.gz")

        # Download if not exists or if hash verification fails
        if not os.path.exists(tar_path) or not verify_sha256(
            tar_path, sha256_hashes[split]
        ):
            print(f"Downloading {split} dataset...")
            urllib.request.urlretrieve(url, tar_path)

            # Verify hash after download
            if not verify_sha256(tar_path, sha256_hashes[split]):
                raise RuntimeError(
                    f"SHA-256 hash verification failed for {split}ing dataset. The file may be corrupted."
                )
            print(f"SHA-256 hash verification successful for {split}ing dataset.")

        # Store paths to the extracted files
        prefix = file_prefixes[split]
        file_paths[split] = {
            "de": os.path.join(root_dir, f"{prefix}.de"),
            "en": os.path.join(root_dir, f"{prefix}.en"),
        }

        # Extract if not already extracted
        if not os.path.exists(file_paths[split]["de"]) or not os.path.exists(
            file_paths[split]["en"]
        ):
            print(f"Extracting {split} dataset...")
            with tarfile.open(tar_path, "r:gz") as tar:
                tar.extractall(path=root_dir)

    return file_paths


def multi30k_loader():
    """Load Multi30k dataset for German-English translation.

    Returns:
        Tuple of (train_dataset, val_dataset, test_dataset)
    """
    # Download and extract dataset
    file_paths = download_multi30k()

    # Create datasets
    train_ds = Multi30kDataset(file_paths["train"]["de"], file_paths["train"]["en"])
    val_ds = Multi30kDataset(file_paths["valid"]["de"], file_paths["valid"]["en"])
    # For now, use validation set as test set since test set is not publicly available
    test_ds = Multi30kDataset(file_paths["valid"]["de"], file_paths["valid"]["en"])

    return train_ds, val_ds, test_ds


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
