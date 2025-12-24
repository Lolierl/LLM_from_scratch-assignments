import os
import argparse
import json
from pathlib import Path
from tqdm import tqdm
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.distributed import DistributedSampler
from transformers import AutoTokenizer
from transformers import AutoModelForCausalLM, AutoTokenizer
from custom_model.model import Transformer, ModelArgs
from src.optim import AdamW, get_lr_cosine_schedule, gradient_clipping
import time
IGNORE_INDEX = -100

class SFTDataset(Dataset):
    def __init__(self, data, tokenizer, max_length=2048):
        self.data = data
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        messages = self.data[idx]["messages"]
        input_ids = self.tokenizer.apply_chat_template(
            messages,
            return_dict=True,
            return_tensors="pt", 
            add_generation_prompt=False, 
            add_special_tokens=False
        )["input_ids"].squeeze(0).long()

        if len(messages) >= 2 and messages[-1]["role"] == "assistant":
            prompt_ids = self.tokenizer.apply_chat_template(
                messages[:-1],
                return_dict=True,
                return_tensors="pt", 
                add_generation_prompt=True, 
                add_special_tokens=False
            )["input_ids"].squeeze(0)
            prompt_len = prompt_ids.shape[0]
        else:
            prompt_len = 0
        if len(input_ids) > self.max_length:
            start = len(input_ids) - self.max_length
            input_ids = input_ids[start:]
            prompt_len = max(0, prompt_len - start)

        labels = torch.full_like(input_ids, IGNORE_INDEX)
        labels[prompt_len:] = input_ids[prompt_len:]

        return {
            "input_ids": input_ids,
            "labels": labels,
        }
def sft_collate_fn(batch, pad_token_id):
    max_len = max(len(x["input_ids"]) for x in batch)

    input_ids, labels = [], []

    for x in batch:
        pad_len = max_len - len(x["input_ids"])

        input_ids.append(
            torch.cat([
                x["input_ids"],
                torch.full((pad_len,), pad_token_id),
            ])
        )

        labels.append(
            torch.cat([
                x["labels"],
                torch.full((pad_len,), IGNORE_INDEX),
            ])
        )

    return {
        "input_ids": torch.stack(input_ids),
        "labels": torch.stack(labels),
    }


def setup_distributed():
    """Setup distributed training environment"""
    rank = int(os.environ.get("RANK", "0"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    backend = 'nccl' if torch.cuda.is_available() else 'gloo'
    dist.init_process_group(backend=backend, rank=rank, world_size=world_size)
    torch.cuda.set_device(local_rank)
    return rank, local_rank, world_size


def build_model(args, model_args, device):
    """Build transformer model from ModelArgs dataclass"""
    model = Transformer(model_args)
    checkpoint = torch.load(args.model, map_location=device)
    state_dict = checkpoint["model_state_dict"]
    from collections import OrderedDict
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        name = k.replace("_orig_mod.", "") if k.startswith("_orig_mod.") else k
        new_state_dict[name] = v
    
    model.load_state_dict(new_state_dict)
    
    model.to(device)
    return model


def save_extended_checkpoint(path, model, optimizer, iteration, rank, world_size, local_rank):
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
    """Load checkpoint and restore all training states"""
    map_location = map_location or "cpu"
    checkpoint = torch.load(path, map_location=map_location)
    model.load_state_dict(checkpoint["model_state_dict"])
    
    if optimizer is not None and "optimizer_state_dict" in checkpoint:
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    
    iteration = checkpoint.get("iteration", 0)
    torch.set_rng_state(checkpoint.get("torch_rng_state", torch.get_rng_state()))
    
    if torch.cuda.is_available() and checkpoint.get("cuda_rng_state_all", None) is not None:
        torch.cuda.set_rng_state_all(checkpoint["cuda_rng_state_all"]) 
    
    return iteration

def evaluate(model, dataloader, device, autocast_ctx):
    model.eval()
    total_loss = 0.0
    total_tokens = 0

    with torch.no_grad():
        for batch in dataloader:
            input_ids = batch["input_ids"].to(device, non_blocking=True)
            labels = batch["labels"].to(device, non_blocking=True)

            with autocast_ctx:
                logits = model(input_ids=input_ids)

                shift_logits = logits[:, :-1, :].contiguous()
                shift_labels = labels[:, 1:].contiguous()

                shift_logits = shift_logits.view(-1, shift_logits.size(-1))
                shift_labels = shift_labels.view(-1)

                mask = shift_labels != IGNORE_INDEX
                if mask.any():
                    loss = torch.nn.functional.cross_entropy(
                        shift_logits[mask],
                        shift_labels[mask],
                        reduction="sum",
                    )
                    total_loss += loss.item()
                    total_tokens += mask.sum().item()

    total_loss_t = torch.tensor(total_loss, device=device)
    total_tokens_t = torch.tensor(total_tokens, device=device)
    dist.all_reduce(total_loss_t, op=dist.ReduceOp.SUM)
    dist.all_reduce(total_tokens_t, op=dist.ReduceOp.SUM)

    model.train()
    return (
        total_loss_t.item() / total_tokens_t.item()
        if total_tokens_t.item() > 0 else 0.0
    )


def make_train_step(
    model,
    optimizer,
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

            shift_logits = logits[:, :-1, :].contiguous()
            shift_labels = labels[:, 1:].contiguous()

            shift_logits = shift_logits.view(-1, shift_logits.size(-1))
            shift_labels = shift_labels.view(-1)

            loss = torch.nn.functional.cross_entropy(
                shift_logits,
                shift_labels,
                ignore_index=IGNORE_INDEX,
                reduction="mean",
            )

        loss_to_backward = loss / accumulation_steps
        loss_to_backward.backward()

        step_state["micro_step"] += 1
        did_step = False

        if step_state["micro_step"] % accumulation_steps == 0:
            gradient_clipping(model.parameters(), grad_clip)
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
            step_state["global_step"] += 1
            did_step = True

        return loss.detach(), did_step, step_state["global_step"]

    return train_step


def train(args, model_args):
    rank, local_rank, world_size = setup_distributed()
    device = torch.device(f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu")

    # ========== Tokenizer ==========
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_path, trust_remote_code=True, enable_thinking=True)
    tokenizer.pad_token = tokenizer.eos_token

    # ========== Dataset ==========
    import datasets
    from pathlib import Path
    
    # Check if it's a datasets directory or parquet files
    data_path = Path(args.train_data_path)
    if (data_path / "dataset_info.json").exists():
        # It's a datasets directory
        full_data = datasets.load_from_disk(args.train_data_path)
    elif (data_path / "data").exists() and list((data_path / "data").glob("*.parquet")):
        # It's a directory with parquet files
        parquet_files = list((data_path / "data").glob("*.parquet"))
        if rank == 0:
            print(f"Loading {len(parquet_files)} parquet files from {data_path / 'data'}")
        full_data = datasets.load_dataset("parquet", data_files=str(parquet_files[0]))['train']
        for pf in parquet_files[1:]:
            full_data = datasets.concatenate_datasets([
                full_data, 
                datasets.load_dataset("parquet", data_files=str(pf))['train']
            ])
    else:
        # Try loading directly as parquet
        full_data = datasets.load_dataset("parquet", data_dir=str(data_path))['train']
    
    # Split: args.val_split_ratio for validation, rest for training
    split_data = full_data.train_test_split(test_size=args.val_split_ratio, seed=42)
    train_data = split_data['train']
    val_data = split_data['test']
    
    if rank == 0:
        print(f"Dataset split: {len(train_data)} train, {len(val_data)} val ({args.val_split_ratio*100:.1f}%)")

    train_dataset = SFTDataset(train_data, tokenizer, max_length=args.context_length)
    val_dataset = SFTDataset(val_data, tokenizer, max_length=args.context_length)

    per_device_batch = args.per_device_batch_size

    # gradient accumulation
    if args.global_batch_size % (per_device_batch * world_size) != 0:
        if rank == 0:
            print("Warning: global_batch_size not divisible; adjusting accumulation steps")
    accumulation_steps = max(1, args.global_batch_size // (per_device_batch * world_size))

    sampler = DistributedSampler(train_dataset, num_replicas=world_size, rank=rank, shuffle=True, drop_last=True)
    
    collate_fn = lambda batch: sft_collate_fn(batch, tokenizer.pad_token_id)
    loader = DataLoader(
        train_dataset, 
        batch_size=per_device_batch, 
        sampler=sampler, 
        num_workers=2, 
        pin_memory=True, 
        drop_last=True,
        collate_fn=collate_fn
    )

    # validation sampler (no shuffle)
    val_sampler = DistributedSampler(val_dataset, num_replicas=world_size, rank=rank, shuffle=False, drop_last=False)
    val_loader = DataLoader(
        val_dataset, 
        batch_size=per_device_batch, 
        sampler=val_sampler, 
        num_workers=2, 
        pin_memory=True, 
        drop_last=False,
        collate_fn=collate_fn
    )
    print("Fetching first batch to test __getitem__...")
    first_batch = next(iter(loader))

    # model
    model = build_model(args, model_args, device)
    model = torch.compile(model)
    
    # optimizer
    optimizer = AdamW(model.parameters(), lr=args.max_lr, weight_decay=args.weight_decay)

    # AMP
    use_amp = torch.cuda.is_available() and args.use_bf16
    if use_amp:
        amp_dtype = torch.bfloat16
        autocast_ctx = torch.amp.autocast(device_type='cuda', dtype=amp_dtype)
        if rank == 0:
            print("Using BF16")
    else:
        autocast_ctx = torch.amp.autocast(device_type='cuda', enabled=False)
        if rank == 0:
            print("Using FP32")

    model = DDP(model, device_ids=[local_rank] if torch.cuda.is_available() else None, find_unused_parameters=False)

    # resume
    start_iter = 0
    if args.resume is not None:
        map_loc = {"cuda:0": f"cuda:{local_rank}"} if torch.cuda.is_available() else "cpu"
        start_iter = load_extended_checkpoint(args.resume, model.module, optimizer, map_location=map_loc)
        if rank == 0:
            print(f"Resumed from {args.resume} at iteration {start_iter}")

    # training parameters
    dataset_size = len(train_dataset)
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

    train_step = make_train_step(
        model=model,
        optimizer=optimizer,
        autocast_ctx=autocast_ctx,
        accumulation_steps=accumulation_steps,
        grad_clip=args.grad_clip,
    )

    model.train()

    for epoch in range(args.epochs):
        sampler.set_epoch(epoch)

        if rank == 0:
            pbar = tqdm(total=len(loader), desc=f"Epoch {epoch}")
        
        for _, batch in enumerate(loader):
            loss, did_step, global_step = train_step(batch, device)
            if did_step:
                lr = get_lr_cosine_schedule(
                    global_step,
                    args.max_lr,
                    args.min_lr,
                    warmup_steps,
                    total_steps,
                )
                for pg in optimizer.param_groups:
                    pg["lr"] = lr

                # logging
                if rank == 0:
                    log_data["train_loss"].append(loss.item())
                    log_data["step"].append(global_step)
                    pbar.set_postfix({"loss": loss.item(), "lr": lr})

                # evaluation
                if global_step > 0 and global_step % args.eval_interval == 0:
                    eval_loss = evaluate(model, val_loader, device, autocast_ctx)
                    if rank == 0:
                        log_data["eval_loss"].append(eval_loss)
                        print(f"\n[Eval @ step {global_step}] loss = {eval_loss:.6f}")

                # checkpoint
                if rank == 0 and global_step > 0 and global_step % args.save_interval == 0:
                    ckpt_path = Path(args.out_dir) / f"checkpoint_step_{global_step}.pt"
                    save_extended_checkpoint(
                        ckpt_path,
                        model.module,
                        optimizer,
                        global_step,
                        rank,
                        world_size,
                        local_rank,
                    )

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
            global_step, 
            rank, 
            world_size, 
            local_rank
        )
        print(f"Saved final checkpoint to {final_ckpt_path}")
        
        # Save logs
        log_path = Path(args.out_dir) / "sft_train_eval_log.json"
        with open(log_path, "w") as f:
            json.dump(log_data, f, indent=2)
        print(f"Saved training log to {log_path}")

    dist.destroy_process_group()


def parse_args():
    p = argparse.ArgumentParser(description="Supervised Fine-Tuning with DDP")
    p.add_argument("--model", default="/thullms/3022377347/hw3_pretrain_fixadam_checkpoints/checkpoint_final.pt", help="Path to HF model directory")
    p.add_argument("--train_data_path", type=str, default="/thullms/public/data/tulu-3-sft-mixture_ctx2048_filtered")
    p.add_argument("--tokenizer_path", type=str, default="/thullms/3022377347/Qwen/Qwen3-0.6B-Base", help="Path to tokenizer (e.g., custom_model_hf)")
    p.add_argument("--val_split_ratio", type=float, default=0.01, help="Ratio of training data to use for validation")
    p.add_argument("--out_dir", type=str, default="/thullms/3022377347/sft_checkpoints")
    p.add_argument("--context_length", type=int, default=2048)

    p.add_argument("--per_device_batch_size", type=int, default=8)
    p.add_argument("--global_batch_size", type=int, default=128)
    p.add_argument("--epochs", type=int, default=1)
    p.add_argument("--max_steps", type=int, default=None)

    p.add_argument("--warmup_ratio", type=float, default=0.01)
    p.add_argument("--max_lr", type=float, default=5e-4)
    p.add_argument("--min_lr", type=float, default=5e-6)
    p.add_argument("--weight_decay", type=float, default=0.1)
    p.add_argument("--grad_clip", type=float, default=1.0)

    p.add_argument("--use_bf16", action="store_true", default=True)
    p.add_argument("--no_bf16", action="store_false", dest="use_bf16")

    p.add_argument("--log_interval", type=int, default=10)
    p.add_argument("--save_interval", type=int, default=500)
    p.add_argument("--eval_interval", type=int, default=500)

    p.add_argument("--resume", type=str, default=None)

    return p.parse_args()


if __name__ == '__main__':
    args = parse_args()
    os.makedirs(args.out_dir, exist_ok=True)
    
    # Model configuration - adjust these to match your pretrained model
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