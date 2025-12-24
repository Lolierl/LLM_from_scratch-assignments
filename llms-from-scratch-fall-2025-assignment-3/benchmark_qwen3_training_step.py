import os
import argparse
import math
import time
import json
from pathlib import Path
from tqdm import tqdm
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler

from custom_model.model import Transformer, ModelArgs
from src.data_module import TokenizedDataset
from src.optim import AdamW, get_lr_cosine_schedule, gradient_clipping
import src.checkpoint as ckpt
from benchmark import benchmark_cuda_fn

def setup_distributed():
    """Setup distributed training environment"""
    rank = int(os.environ.get("RANK", "0"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    backend = 'nccl' if torch.cuda.is_available() else 'gloo'
    dist.init_process_group(backend=backend, rank=rank, world_size=world_size)
    torch.cuda.set_device(local_rank)
    return rank, local_rank, world_size


def build_model(model_args, device):
    """Build transformer model from ModelArgs dataclass"""
    model = Transformer(model_args)
    model.to(device)
    return model


def save_extended_checkpoint(path, model, optimizer, scaler, iteration, rank, world_size, local_rank):
    """Save checkpoint with all training states including AMP scaler"""
    checkpoint = {
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scaler_state_dict": scaler.state_dict(),
        "iteration": iteration,
        "torch_rng_state": torch.get_rng_state(),
        "cuda_rng_state_all": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
        "distributed": {
            "world_size": world_size,
            "rank": rank,
            "local_rank": local_rank,
        }
    }
    torch.save(checkpoint, path)


def load_extended_checkpoint(path, model, optimizer=None, scaler=None, map_location=None):
    """Load checkpoint and restore all training states"""
    map_location = map_location or "cpu"
    checkpoint = torch.load(path, map_location=map_location)
    model.load_state_dict(checkpoint["model_state_dict"])
    
    if optimizer is not None and "optimizer_state_dict" in checkpoint:
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    
    if scaler is not None and "scaler_state_dict" in checkpoint:
        scaler.load_state_dict(checkpoint["scaler_state_dict"])
    
    iteration = checkpoint.get("iteration", 0)
    torch.set_rng_state(checkpoint.get("torch_rng_state", torch.get_rng_state()))
    
    if torch.cuda.is_available() and checkpoint.get("cuda_rng_state_all", None) is not None:
        torch.cuda.set_rng_state_all(checkpoint["cuda_rng_state_all"]) 
    
    return iteration


def evaluate(model, dataloader, device, autocast_ctx):
    """Compute evaluation loss on validation dataset.
       All ranks participate in evaluation and synchronize results.
    """
    model.eval()
    total_loss = 0.0
    total_tokens = 0
    
    with torch.no_grad():
        for batch in dataloader:
            input_ids = batch["input_ids"].to(device, non_blocking=True)
            labels = batch["labels"].to(device, non_blocking=True)

            with autocast_ctx:
                logits = model(input_ids=input_ids)
                logits_flat = logits.view(-1, logits.size(-1))
                labels_flat = labels.view(-1).long()
                loss = torch.nn.functional.cross_entropy(logits_flat, labels_flat, reduction="mean")

            # Accumulate loss and token count for this batch
            batch_loss = loss.detach() * input_ids.numel()
            batch_tokens = input_ids.numel()
            
            total_loss += batch_loss.item()
            total_tokens += batch_tokens

    # Gather total loss and tokens across all GPUs
    total_loss_tensor = torch.tensor(total_loss, device=device)
    total_tokens_tensor = torch.tensor(total_tokens, device=device)
    
    dist.all_reduce(total_loss_tensor, op=dist.ReduceOp.SUM)
    dist.all_reduce(total_tokens_tensor, op=dist.ReduceOp.SUM)

    model.train()
    
    avg_loss = total_loss_tensor.item() / total_tokens_tensor.item() if total_tokens_tensor.item() > 0 else 0.0
    return avg_loss

def make_train_step(
    model,
    optimizer,
    scaler,
    autocast_ctx,
    accumulation_steps: int,
    grad_clip: float,
):
    step_state = {
        "micro_step": 0,
        "global_step": 0,
    }

    def train_step(batch, device):
        """
        One micro training step.
        Returns:
            loss (unscaled, for logging)
            did_step (bool): whether optimizer.step() happened
        """
        input_ids = batch["input_ids"].to(device, non_blocking=True)
        labels = batch["labels"].to(device, non_blocking=True)

        with autocast_ctx:
            logits = model(input_ids=input_ids)
            logits_flat = logits.view(-1, logits.size(-1))
            labels_flat = labels.view(-1).long()
            loss = torch.nn.functional.cross_entropy(
                logits_flat, labels_flat, reduction="mean"
            )

        loss_to_backward = loss / accumulation_steps
        scaler.scale(loss_to_backward).backward()

        step_state["micro_step"] += 1
        did_step = False

        if step_state["micro_step"] % accumulation_steps == 0:
            scaler.unscale_(optimizer)
            gradient_clipping(model.parameters(), grad_clip)

            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)

            step_state["global_step"] += 1
            did_step = True

        return loss.detach(), did_step, step_state["global_step"]

    return train_step
def train(args, model_args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    accumulation_steps = max(1, args.global_batch_size // args.per_device_batch_size)
    # model
    model = build_model(model_args, device)
    model = torch.compile(model, mode="reduce-overhead")
    # optimizer
    optimizer = AdamW(model.parameters(), lr=args.max_lr, weight_decay=args.weight_decay)

    # AMP - Fixed deprecation warning
    use_amp = torch.cuda.is_available() and args.use_bf16
    if use_amp:
        amp_dtype = torch.bfloat16
        autocast_ctx = torch.amp.autocast(device_type='cuda', dtype=amp_dtype)
        scaler = torch.amp.GradScaler('cuda', enabled=args.use_bf16)
        print("Using BF16")
    else:
        autocast_ctx = torch.amp.autocast(device_type='cuda', enabled=False)
        scaler = torch.amp.GradScaler('cuda', enabled=False)
        print("Using FP32")

    start_iter = 0
    train_step = make_train_step(
        model=model,
        optimizer=optimizer,
        scaler=scaler,
        autocast_ctx=autocast_ctx,
        accumulation_steps=accumulation_steps,
        grad_clip=args.grad_clip,
    )

    model.train()    
    vocab_size = model_args.vocab_size
    B = 8
    L = 2048
    batch = {
        "input_ids": torch.randint(0, vocab_size, (B, L), device=device),
        "labels": torch.randint(0, vocab_size, (B, L), device=device),
    }

    def step_fn():
        torch.compiler.cudagraph_mark_step_begin()
        for _ in range(accumulation_steps):
            loss, did_step, global_step = train_step(batch, device)
    
    train_step = make_train_step(model, optimizer, scaler, autocast_ctx, accumulation_steps, args.grad_clip)
    mean_ms, std_ms = benchmark_cuda_fn(step_fn, n_warmup=5, n_runs=20)
    print(f"Training step time (B={B}, L={L}): {mean_ms:.2f} ± {std_ms:.2f} ms")


def parse_args():
    p = argparse.ArgumentParser(description="Distributed training with eval & logs")

    p.add_argument("--bin_path", type=str, default="/thullms/3022377347/tokenized_fineweb_edu_10b/train.bin")
    p.add_argument("--val_bin_path", type=str, default="/thullms/3022377347/tokenized_fineweb_edu_10b/val.bin")
    p.add_argument("--out_dir", type=str, default="/thullms/3022377347/checkpoints")
    p.add_argument("--context_length", type=int, default=2048)

    p.add_argument("--per_device_batch_size", type=int, default=8)
    p.add_argument("--global_batch_size", type=int, default=256)
    p.add_argument("--epochs", type=int, default=1)
    p.add_argument("--max_steps", type=int, default=None)

    p.add_argument("--warmup_ratio", type=float, default=0.01)
    p.add_argument("--max_lr", type=float, default=3e-3)
    p.add_argument("--min_lr", type=float, default=3e-5)
    p.add_argument("--weight_decay", type=float, default=0.1)
    p.add_argument("--grad_clip", type=float, default=1.0)

    p.add_argument("--use_bf16", action="store_true", default=True)
    p.add_argument("--no_bf16", action="store_false", dest="use_bf16")

    p.add_argument("--log_interval", type=int, default=10)
    p.add_argument("--save_interval", type=int, default=200)
    p.add_argument("--eval_interval", type=int, default=200)

    p.add_argument("--resume", type=str, default=None)

    return p.parse_args()


if __name__ == '__main__':
    args = parse_args()
    os.makedirs(args.out_dir, exist_ok=True)
    
    # Model configuration
    model_args = ModelArgs(
        vocab_size=151936,
        embed_dim=1024,
        inter_dim=3072,
        num_attention_heads=16,
        num_key_value_heads=8,
        head_dim=128,
        n_layers=28,
        rope_theta=10000.0,
    )
    
    train(args, model_args)