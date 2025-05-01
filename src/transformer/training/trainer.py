import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.optim.lr_scheduler import LambdaLR
import os

from transformer.model.transformer import make_model
from transformer.data.dataset import (
    load_tokenizers,
    create_dataloaders,
    build_vocab,
    tokenize,
)
from transformer.training.util import (
    TrainState,
    LabelSmoothing,
    SimpleLossCompute,
    rate,
    run_epoch,
)
from transformer.config.config import TranslationConfig


def ensure_save_dir(cfg: TranslationConfig) -> str:
    """Create and return the path to the save directory."""
    save_dir = os.path.join("checkpoints", cfg.file_prefix)
    os.makedirs(save_dir, exist_ok=True)
    return save_dir


def build_vocabularies(cfg: TranslationConfig):
    tokenizers = load_tokenizers(cfg.spacy_models)
    # Create datasets
    train_ds, val_ds, test_ds = cfg.dataset_loader()

    # Build vocabularies
    src_vocab = build_vocab(
        (src for dataset in [train_ds, val_ds, test_ds] for src, _ in dataset),
        lambda x: tokenize(x, tokenizers[cfg.src_lang]),
        min_freq=2,
        specials=["<blank>", "<s>", "</s>", "<unk>"],
    )
    src_vocab.set_default_index(src_vocab["<unk>"])

    train_ds, val_ds, test_ds = cfg.dataset_loader()
    tgt_vocab = build_vocab(
        (tgt for dataset in [train_ds, val_ds, test_ds] for _, tgt in dataset),
        lambda x: tokenize(x, tokenizers[cfg.tgt_lang]),
        min_freq=2,
        specials=["<blank>", "<s>", "</s>", "<unk>"],
    )
    tgt_vocab.set_default_index(tgt_vocab["<unk>"])

    return src_vocab, tgt_vocab


def setup_distributed(rank: int, world_size: int, seed: int):
    dist.init_process_group(
        backend="nccl", init_method="env://", world_size=world_size, rank=rank
    )
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)

def cleanup_distributed():
    dist.destroy_process_group()

def train_worker(
    rank: int,
    world_size: int,
    cfg: TranslationConfig,
    seed: int = 42,
):
    """Worker function for distributed training."""
    if cfg.distributed:
        setup_distributed(rank, world_size, seed)

    # Load tokenizers
    tokenizers = load_tokenizers(cfg.spacy_models)

    # Build vocabularies
    src_vocab, tgt_vocab = build_vocabularies(cfg)

    # Create model
    model = make_model(
        len(src_vocab),
        len(tgt_vocab),
        N=cfg.model_layers,
        d_model=cfg.d_model,
        d_ff=cfg.d_ff,
        h=cfg.h,
        dropout=cfg.dropout,
    )
    # Move model to GPU
    device = f"cuda:{rank}"
    model.to(device)
    module = model
    if cfg.distributed:
        model = DDP(model, device_ids=[rank])
        module = model.module

    # Create optimizer and scheduler
    optimizer = torch.optim.Adam(
        model.parameters(), lr=cfg.base_lr, betas=(0.9, 0.98), eps=1e-9
    )
    scheduler = LambdaLR(
        optimizer,
        lambda step: rate(step, cfg.d_model, factor=1, warmup=cfg.warmup),
    )

    # Create loss function
    criterion = LabelSmoothing(
        size=len(tgt_vocab), padding_idx=tgt_vocab["<blank>"], smoothing=cfg.smoothing
    )
    criterion.to(device)
    loss_compute = SimpleLossCompute(module.generator, criterion)

    # Create dataloaders
    train_dl, val_dl, test_dl = create_dataloaders(
        cfg,
        tokenizers,
        src_vocab,
        tgt_vocab,
        device=f"cuda:{rank}",
        max_len=cfg.max_len,
        distributed=cfg.distributed,
    )

    # Training loop
    train_state = TrainState()
    is_main_process = rank == 0 or not cfg.distributed
    torch.cuda.empty_cache()

    # Create save directory
    save_dir = ensure_save_dir(cfg)

    for epoch in range(cfg.num_epochs):
        if cfg.distributed:
            train_dl.sampler.set_epoch(epoch)
            val_dl.sampler.set_epoch(epoch)

        model.train()
        if rank == 0:
            print(f"==== Epoch {epoch} Training ====", flush=True)

        _, train_state = run_epoch(
            train_dl,
            model,
            loss_compute,
            optimizer,
            scheduler,
            rank,
            cfg.distributed,
            mode="train",
            accum_iter=cfg.accum_iter,
            train_state=train_state,
        )
        torch.cuda.empty_cache()

        if rank == 0:
            print(f"==== Epoch {epoch} Validation ====", flush=True)
        model.eval()
        with torch.no_grad():
            val_loss, _ = run_epoch(
                val_dl,
                model,
                loss_compute,
                optimizer,
                scheduler,
                rank,
                cfg.distributed,
                mode="eval",
            )
            if cfg.distributed:
                dist.reduce(val_loss, dst=0)
            if rank == 0:
                print(f"Validation Loss: {val_loss}")
                print("")


        torch.cuda.empty_cache()
        # Save checkpoint
        if is_main_process:
            checkpoint = {
                "model": module.state_dict(),
                "optimizer": optimizer.state_dict(),
                "scheduler": scheduler.state_dict(),
                "train_state": train_state,
                "epoch": epoch,
                "val_loss": val_loss,
                "src_vocab": src_vocab,
                "tgt_vocab": tgt_vocab,
            }
            checkpoint_path = os.path.join(save_dir, f"epoch_{epoch:02d}.pt")
            torch.save(checkpoint, checkpoint_path)

    # Save final model
    if is_main_process:
        final_path = os.path.join(save_dir, "final.pt")
        torch.save(module.state_dict(), final_path)

    if cfg.distributed:
        cleanup_distributed()


def train_model(cfg: TranslationConfig):
    """Main training function."""
    if cfg.distributed:
        ngpus_per_node = torch.cuda.device_count()
        print(f"world size: {ngpus_per_node}")
        os.environ["MASTER_ADDR"] = "localhost"
        os.environ["MASTER_PORT"] = cfg.master_port
        mp.spawn(train_worker, nprocs=ngpus_per_node, args=(ngpus_per_node, cfg))
    else:
        train_worker(0, 1, cfg)


def load_trained_model(cfg: TranslationConfig, path: str):
    """Load a trained model from checkpoint."""
    checkpoint = torch.load(path, weights_only=False)

    # Load vocabularies
    src_vocab, tgt_vocab = checkpoint["src_vocab"], checkpoint["tgt_vocab"]

    model = make_model(
        len(src_vocab),
        len(tgt_vocab),
        N=cfg.model_layers,
        d_model=cfg.d_model,
        d_ff=cfg.d_ff,
        h=cfg.h,
        dropout=cfg.dropout,
    )

    model.load_state_dict(checkpoint["model"])

    return model, src_vocab, tgt_vocab
