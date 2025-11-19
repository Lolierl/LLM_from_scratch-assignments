import os
import argparse
import math
import time
from pathlib import Path

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler

from custom_model.model import Transformer, ModelArgs
from src.data_module import TokenizedDataset
from src.optim import AdamW, get_lr_cosine_schedule, gradient_clipping
import src.checkpoint as ckpt


def setup_distributed():
    # Expect environment variables set by launcher (torchrun / torch.distributed.launch)
    rank = int(os.environ.get("RANK", "0"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    backend = 'nccl' if torch.cuda.is_available() else 'gloo'
    dist.init_process_group(backend=backend, rank=rank, world_size=world_size)
    torch.cuda.set_device(local_rank)
    return rank, local_rank, world_size


def build_model(args, device):
    model_args = ModelArgs(
        vocab_size=args.vocab_size,
        embed_dim=args.embed_dim,
        inter_dim=args.inter_dim,
        num_attention_heads=args.num_attention_heads,
        num_key_value_heads=args.num_key_value_heads,
        head_dim=args.head_dim,
        n_layers=args.n_layers,
        rope_theta=args.rope_theta,
    )
    model = Transformer(model_args)
    model.to(device)
    return model


def save_extended_checkpoint(path, model, optimizer, iteration, rank, world_size, local_rank):
    # include RNG states and distributed metadata so a resumed run can restore exact state
    checkpoint = {
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
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


def load_extended_checkpoint(path, model, optimizer=None, map_location=None):
    map_location = map_location or ("cpu")
    checkpoint = torch.load(path, map_location=map_location)
    model.load_state_dict(checkpoint["model_state_dict"])
    if optimizer is not None and "optimizer_state_dict" in checkpoint:
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    iteration = checkpoint.get("iteration", 0)
    torch.set_rng_state(checkpoint.get("torch_rng_state", torch.get_rng_state()))
    if torch.cuda.is_available() and checkpoint.get("cuda_rng_state_all", None) is not None:
        torch.cuda.set_rng_state_all(checkpoint["cuda_rng_state_all"]) 
    return iteration


def train(args, model_args):
    rank, local_rank, world_size = setup_distributed()
    device = torch.device(f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu")

    # dataset and sampler
    dataset = TokenizedDataset(args.bin_path, max_length=args.context_length)
    per_device_batch = args.per_device_batch_size
    # compute gradient accumulation steps to reach global batch size
    if args.global_batch_size % (per_device_batch * world_size) != 0:
        if rank == 0:
            print("Warning: global_batch_size not divisible by per-device * world_size; adjusting accumulation steps")
    accumulation_steps = max(1, args.global_batch_size // (per_device_batch * world_size))

    sampler = DistributedSampler(dataset, num_replicas=world_size, rank=rank, shuffle=True)
    loader = DataLoader(dataset, batch_size=per_device_batch, sampler=sampler, drop_last=True, num_workers=2, pin_memory=True)

    # model
    model = build_model(model_args, device)

    # optimizer
    optimizer = AdamW(model.parameters(), lr=args.max_lr, weight_decay=args.weight_decay)

    # wrap DDP
    model = DDP(model, device_ids=[local_rank] if torch.cuda.is_available() else None)

    # optionally resume
    start_iter = 0
    if args.resume is not None:
        # load on all ranks so optimizer states are consistent locally
        map_loc = {"cuda:0": f"cuda:{local_rank}"} if torch.cuda.is_available() else "cpu"
        start_iter = load_extended_checkpoint(args.resume, model.module if isinstance(model, DDP) else model, optimizer=optimizer, map_location=map_loc)
        if rank == 0:
            print(f"Resumed from {args.resume} at iteration {start_iter}")

    # training loop params
    dataset_size = len(dataset)
    steps_per_epoch = dataset_size // args.global_batch_size if args.global_batch_size > 0 else len(loader)
    total_steps = args.epochs * max(1, steps_per_epoch)
    warmup_steps = max(1, int(total_steps * args.warmup_ratio))

    # main loop
    model.train()
    global_step = start_iter
    for epoch in range(args.epochs):
        sampler.set_epoch(epoch)
        for step, batch in enumerate(loader):
            # forward
            input_ids = batch["input_ids"].to(device, non_blocking=True)
            labels = batch["labels"].to(device, non_blocking=True)

            logits = model(input_ids=input_ids)
            # logits: (B, L, V)
            logits = logits.view(-1, logits.size(-1))
            labels_flat = labels.view(-1)
            loss = torch.nn.functional.cross_entropy(logits, labels_flat, reduction='mean')

            loss = loss / accumulation_steps
            loss.backward()

            if (step + 1) % accumulation_steps == 0:
                # update learning rate
                lr = get_lr_cosine_schedule(global_step, args.max_lr, args.min_lr, warmup_steps, total_steps)
                for pg in optimizer.param_groups:
                    pg['lr'] = lr

                # gradient clipping
                gradient_clipping(model.parameters(), args.grad_clip)

                optimizer.step()
                optimizer.zero_grad()

                if rank == 0 and global_step % args.log_interval == 0:
                    print(f"Iter {global_step:8d} epoch {epoch} loss={loss.item()*accumulation_steps:.6f} lr={lr:.6e}")

                # checkpoint
                if rank == 0 and global_step % args.save_interval == 0:
                    out_path = Path(args.out_dir) / f"ckpt_iter_{global_step}.pt"
                    save_extended_checkpoint(out_path, model.module if isinstance(model, DDP) else model, optimizer, global_step, rank, world_size, local_rank)

                global_step += 1

            # early stop
            if args.max_steps is not None and global_step >= args.max_steps:
                break
        if args.max_steps is not None and global_step >= args.max_steps:
            break

    # final checkpoint
    if rank == 0:
        out_path = Path(args.out_dir) / f"ckpt_final_iter_{global_step}.pt"
        save_extended_checkpoint(out_path, model.module if isinstance(model, DDP) else model, optimizer, global_step, rank, world_size, local_rank)

    # cleanup
    dist.destroy_process_group()


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--bin_path", type=str, required=True)
    p.add_argument("--out_dir", type=str, default="checkpoints")
    p.add_argument("--context_length", type=int, default=1024)
    p.add_argument("--per_device_batch_size", type=int, default=4)
    p.add_argument("--global_batch_size", type=int, default=512)
    p.add_argument("--epochs", type=int, default=1)
    p.add_argument("--max_steps", type=int, default=None)
    p.add_argument("--warmup_ratio", type=float, default=0.01)
    p.add_argument("--max_lr", type=float, default=3e-3)
    p.add_argument("--min_lr", type=float, default=3e-5)
    p.add_argument("--weight_decay", type=float, default=0.1)
    p.add_argument("--grad_clip", type=float, default=1.0)
    p.add_argument("--log_interval", type=int, default=10)
    p.add_argument("--save_interval", type=int, default=100)
    p.add_argument("--resume", type=str, default=None)
    return p.parse_args()


if __name__ == '__main__':
    args = parse_args()
    os.makedirs(args.out_dir, exist_ok=True)
    model_args = {
        "vocab_size": 151936,
        "embed_dim": 512,
        "inter_dim": 1024,
        "num_attention_heads": 8,
        "num_key_value_heads": 4,
        "head_dim": 64,
        "n_layers": 12,
        "rope_theta": 10000.0,
    }
    train(args, model_args)
