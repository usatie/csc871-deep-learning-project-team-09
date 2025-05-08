import os
import json
import time
from datetime import datetime

import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.optim.lr_scheduler import LambdaLR

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
from transformer.config import (
    TranslationConfig,
    get_checkpoint_path,
    get_final_checkpoint_path,
    get_last_checkpoint,
    get_checkpoint_dir,
)


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


def save_training_metrics(
    save_dir: str,
    epoch: int,
    train_loss: float,
    val_loss: float,
    elapsed: float,
    cfg: TranslationConfig,
):
    """Save training metrics and hyperparameters to a JSON file."""
    metrics_filename = (
        f"training_metrics_bs{cfg.batch_size}_acc{cfg.accum_iter}_warm{cfg.warmup}.json"
    )
    metrics_path = os.path.join(save_dir, metrics_filename)

    # Get current metrics
    current_metrics = {
        "epoch": epoch,
        "train_loss": train_loss,
        "val_loss": val_loss,
        "timestamp": datetime.now().isoformat(),
        "elapsed": elapsed,
    }

    # Get hyperparameters from config
    hyperparams = {
        "model_layers": cfg.model_layers,
        "d_model": cfg.d_model,
        "d_ff": cfg.d_ff,
        "h": cfg.h,
        "dropout": cfg.dropout,
        "smoothing": cfg.smoothing,
        "max_len": cfg.max_len,
        "num_epochs": cfg.num_epochs,
        "batch_size": cfg.batch_size,
        "accum_iter": cfg.accum_iter,
        "warmup": cfg.warmup,
    }

    # Load existing data if file exists
    if os.path.exists(metrics_path):
        with open(metrics_path, "r") as f:
            data = json.load(f)
    else:
        data = {"hyperparameters": hyperparams, "metrics": []}

    # Append new metrics
    data["metrics"].append(current_metrics)

    # Save updated data
    with open(metrics_path, "w") as f:
        json.dump(data, f, indent=2)


def train_worker(
    rank: int,
    world_size: int,
    cfg: TranslationConfig,
    seed: int = 42,
):
    """Worker function for distributed training."""
    if cfg.distributed:
        setup_distributed(rank, world_size, seed)
        # Set device for this process
        torch.cuda.set_device(rank)

    last_checkpoint = get_last_checkpoint(cfg)
    checkpoint = None
    if last_checkpoint:
        last_checkpoint_path = os.path.join(get_checkpoint_dir(cfg), last_checkpoint)
        checkpoint = torch.load(last_checkpoint_path, weights_only=False)
        src_vocab, tgt_vocab = checkpoint["src_vocab"], checkpoint["tgt_vocab"]
    else:
        # Build vocabularies
        src_vocab, tgt_vocab = build_vocabularies(cfg)

    # Load tokenizers
    tokenizers = load_tokenizers(cfg.spacy_models)

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

    if checkpoint:
        model.load_state_dict(checkpoint["model"])

    # Move model to GPU
    device = f"cuda:{rank}"
    model.to(device)
    module = model
    if cfg.distributed:
        model = DDP(model, device_ids=[rank])
        module = model.module

    # Create optimizer and scheduler
    BASE_LEARNING_RATE = 1  # Learning rate is set by scheduler
    optimizer = torch.optim.Adam(
        model.parameters(), lr=BASE_LEARNING_RATE, betas=(0.9, 0.98), eps=1e-9
    )
    if checkpoint:
        optimizer.load_state_dict(checkpoint["optimizer"])
    scheduler = LambdaLR(
        optimizer,
        lambda step: rate(step, cfg.d_model, factor=1, warmup=cfg.warmup),
    )
    if checkpoint:
        scheduler.load_state_dict(checkpoint["scheduler"])

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
    train_state = checkpoint["train_state"] if checkpoint else TrainState()
    is_main_process = rank == 0 or not cfg.distributed
    torch.cuda.empty_cache()

    # Create save directory
    save_dir = ensure_save_dir(cfg)

    if checkpoint:
        start_epoch = checkpoint["epoch"] + 1
    else:
        start_epoch = 1

    if is_main_process:
        if checkpoint:
            print(f"Training from last checkpoint at {last_checkpoint_path}")
        print(f"Start training from epoch {start_epoch}")

    # Synchronize all processes before starting training
    if cfg.distributed:
        dist.barrier()

    for epoch in range(start_epoch, cfg.num_epochs + 1):
        start = time.time()
        if cfg.distributed:
            train_dl.sampler.set_epoch(epoch)
            val_dl.sampler.set_epoch(epoch)

        model.train()
        if is_main_process:
            print(f"==== Epoch {epoch} Training ====", flush=True)

        train_loss, train_state = run_epoch(
            data_iter=train_dl,
            model=model,
            loss_compute=loss_compute,
            optimizer=optimizer,
            scheduler=scheduler,
            rank=rank,
            distributed=cfg.distributed,
            mode="train",
            accum_iter=cfg.accum_iter,
            train_state=train_state,
        )
        torch.cuda.empty_cache()

        if is_main_process:
            # Training loss is already reduced by run_epoch function
            print(f"Training Loss: {train_loss}", flush=True)
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
            # Validation loss is already reduced by run_epoch function
            if is_main_process:
                print(f"Validation Loss: {val_loss}")
                print("")

        # Save metrics and checkpoint
        elapsed = time.time() - start
        if is_main_process:
            # Save metrics with hyperparameters
            save_training_metrics(
                save_dir, epoch, train_loss.item(), val_loss.item(), elapsed, cfg
            )
            # Save checkpoint
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
            # Create checkpoint filename with hyperparameter information
            checkpoint_path = get_checkpoint_path(cfg, epoch)
            torch.save(checkpoint, checkpoint_path)
        if cfg.distributed:
            dist.barrier()
        torch.cuda.empty_cache()

    # Save final model
    if is_main_process:
        final_path = get_final_checkpoint_path(cfg)
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
    print_vocab_stats(src_vocab, tgt_vocab)
    print_model_stats(model)

    model.load_state_dict(checkpoint["model"])

    return model, src_vocab, tgt_vocab


def print_vocab_stats(src_vocab, tgt_vocab):
    """Print vocabulary statistics."""
    print(f"\n{'='*50}")
    print(f"VOCABULARY SUMMARY:")
    print(f"{'='*50}")
    print(f"Source vocabulary: {len(src_vocab):,} tokens")
    print(f"Target vocabulary: {len(tgt_vocab):,} tokens")


def print_model_stats(model):
    """Print model statistics, including parameter counts and memory usage."""
    # Print overall model statistics
    total_params = sum(p.numel() for p in model.parameters())
    total_size_mb = sum(p.numel() * p.element_size() for p in model.parameters()) / (
        1024 * 1024
    )

    print(f"\n{'='*50}")
    print(f"MODEL SUMMARY:")
    print(f"{'='*50}")
    print(f"Total parameters: {total_params:,}")
    print(f"Total size: {total_size_mb:.2f} MB")
    print(f"{'-'*50}")

    # Group parameters by layer or component
    groups = {}
    for name, param in model.named_parameters():
        parts = name.split(".")
        group = ".".join(parts[:3])

        if group not in groups:
            groups[group] = {"params": 0, "size_mb": 0}

        groups[group]["params"] += param.numel()
        groups[group]["size_mb"] += param.numel() * param.element_size() / (1024 * 1024)

    # Print statistics for each group
    print(f"PARAMETER DISTRIBUTION:")
    for group in sorted(groups.keys()):
        params = groups[group]["params"]
        size = groups[group]["size_mb"]
        percent = params / total_params * 100
        print(f"  {group:<30}: {params:,} params ({percent:.1f}%), {size:.2f} MB")

    print(f"{'='*50}")
