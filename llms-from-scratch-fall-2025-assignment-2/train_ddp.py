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


def train(args, model_args):
    rank, local_rank, world_size = setup_distributed()
    device = torch.device(f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu")

    # ========== Dataset ==========
    dataset = TokenizedDataset(args.bin_path, max_length=args.context_length)
    val_dataset = TokenizedDataset(args.val_bin_path, max_length=args.context_length)

    per_device_batch = args.per_device_batch_size

    # gradient accumulation
    if args.global_batch_size % (per_device_batch * world_size) != 0:
        if rank == 0:
            print("Warning: global_batch_size not divisible; adjusting accumulation steps")
    accumulation_steps = max(1, args.global_batch_size // (per_device_batch * world_size))

    sampler = DistributedSampler(dataset, num_replicas=world_size, rank=rank, shuffle=True, drop_last=True)
    loader = DataLoader(dataset, batch_size=per_device_batch, sampler=sampler, num_workers=2, pin_memory=True, drop_last=True)

    # validation sampler (no shuffle)
    val_sampler = DistributedSampler(val_dataset, num_replicas=world_size, rank=rank, shuffle=False, drop_last=False)
    val_loader = DataLoader(val_dataset, batch_size=per_device_batch, sampler=val_sampler, num_workers=2, pin_memory=True, drop_last=False)

    # model
    model = build_model(model_args, device)

    # optimizer
    optimizer = AdamW(model.parameters(), lr=args.max_lr, weight_decay=args.weight_decay)

    # AMP - Fixed deprecation warning
    use_amp = torch.cuda.is_available() and args.use_bf16
    if use_amp:
        amp_dtype = torch.bfloat16
        autocast_ctx = torch.amp.autocast(device_type='cuda', dtype=amp_dtype)
        scaler = torch.amp.GradScaler('cuda', enabled=args.use_bf16)
        if rank == 0:
            print("Using BF16")
    else:
        autocast_ctx = torch.amp.autocast(device_type='cuda', enabled=False)
        scaler = torch.amp.GradScaler('cuda', enabled=False)
        if rank == 0:
            print("Using FP32")

    model = DDP(model, device_ids=[local_rank] if torch.cuda.is_available() else None, find_unused_parameters=False)

    # resume
    start_iter = 0
    if args.resume is not None:
        map_loc = {"cuda:0": f"cuda:{local_rank}"} if torch.cuda.is_available() else "cpu"
        start_iter = load_extended_checkpoint(args.resume, model.module, optimizer, scaler, map_location=map_loc)
        if rank == 0:
            print(f"Resumed from {args.resume} at iteration {start_iter}")

    # training parameters
    dataset_size = len(dataset)
    steps_per_epoch = dataset_size // args.global_batch_size
    total_steps = args.epochs * max(1, steps_per_epoch)
    warmup_steps = max(1, int(total_steps * args.warmup_ratio))

    if rank == 0:
        print(f"Training steps: {total_steps}, warmup: {warmup_steps}")
        print(f"Accumulation steps: {accumulation_steps}")

    # ========== storage for plotting later ==========
    log_data = {
        "train_loss": [],
        "eval_loss": [],
        "step": []
    }

    model.train()
    global_step = start_iter
    optimizer.zero_grad()

    for epoch in range(args.epochs):
        sampler.set_epoch(epoch)

        # progress bar only on rank 0
        if rank == 0:
            pbar = tqdm(total=len(loader), desc=f"Epoch {epoch}")

        for step, batch in enumerate(loader):
            input_ids = batch["input_ids"].to(device, non_blocking=True)
            labels = batch["labels"].to(device, non_blocking=True)

            with autocast_ctx:
                logits = model(input_ids=input_ids)
                logits_flat = logits.view(-1, logits.size(-1))
                labels_flat = labels.view(-1).long()
                loss = torch.nn.functional.cross_entropy(logits_flat, labels_flat, reduction='mean')

            loss = loss / accumulation_steps
            scaler.scale(loss).backward()

            if (step + 1) % accumulation_steps == 0:
                lr = get_lr_cosine_schedule(global_step, args.max_lr, args.min_lr, warmup_steps, total_steps)
                for pg in optimizer.param_groups:
                    pg['lr'] = lr

                scaler.unscale_(optimizer)
                gradient_clipping(model.parameters(), args.grad_clip)

                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()

                # logging
                if rank == 0:
                    actual_loss = loss.item() * accumulation_steps
                    log_data["train_loss"].append(actual_loss)
                    log_data["step"].append(global_step)
                    pbar.set_postfix({"loss": actual_loss, "lr": lr})

                # evaluation - ALL ranks participate
                if global_step > 0 and global_step % args.eval_interval == 0:
                    eval_loss = evaluate(model, val_loader, device, autocast_ctx)
                    if rank == 0:
                        log_data["eval_loss"].append(eval_loss)
                        print(f"\n[Eval @ step {global_step}] loss = {eval_loss:.6f}")

                # checkpoint saving - only rank 0 saves
                if rank == 0 and global_step > 0 and global_step % args.save_interval == 0:
                    ckpt_path = Path(args.out_dir) / f"checkpoint_step_{global_step}.pt"
                    save_extended_checkpoint(
                        ckpt_path, 
                        model.module, 
                        optimizer, 
                        scaler, 
                        global_step, 
                        rank, 
                        world_size, 
                        local_rank
                    )
                    print(f"Saved checkpoint to {ckpt_path}")

                global_step += 1

            if rank == 0:
                pbar.update(1)

            if args.max_steps is not None and global_step >= args.max_steps:
                break

        if rank == 0:
            pbar.close()

        if args.max_steps is not None and global_step >= args.max_steps:
            break

    # Final evaluation
    if rank == 0:
        print("\nRunning final evaluation...")
    final_eval_loss = evaluate(model, val_loader, device, autocast_ctx)
    if rank == 0:
        log_data["eval_loss"].append(final_eval_loss)
        print(f"Final eval loss: {final_eval_loss:.6f}")

    # save final checkpoint and logs
    if rank == 0:
        # Save final checkpoint
        final_ckpt_path = Path(args.out_dir) / "checkpoint_final.pt"
        save_extended_checkpoint(
            final_ckpt_path, 
            model.module, 
            optimizer, 
            scaler, 
            global_step, 
            rank, 
            world_size, 
            local_rank
        )
        print(f"Saved final checkpoint to {final_ckpt_path}")
        
        # Save logs
        log_path = Path(args.out_dir) / "train_eval_log.json"
        with open(log_path, "w") as f:
            json.dump(log_data, f, indent=2)
        print(f"Saved training log to {log_path}")

    dist.destroy_process_group()


def parse_args():
    p = argparse.ArgumentParser(description="Distributed training with eval & logs")

    p.add_argument("--bin_path", type=str, default="/thullms/3022377347/tokenized_fineweb_edu_10b/train.bin")
    p.add_argument("--val_bin_path", type=str, default="/thullms/3022377347/tokenized_fineweb_edu_10b/val.bin")
    p.add_argument("--out_dir", type=str, default="/thullms/3022377347/checkpoints")
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
        embed_dim=512,
        inter_dim=1024,
        num_attention_heads=8,
        num_key_value_heads=4,
        head_dim=64,
        n_layers=12,
        rope_theta=10000.0,
    )
    
    train(args, model_args)