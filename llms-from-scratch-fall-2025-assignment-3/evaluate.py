from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import List, Tuple

import pandas as pd
import torch
import torch.nn.functional as F
from datasets import Dataset, load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
from tqdm import tqdm

os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
os.environ.setdefault("HF_DATASETS_OFFLINE", "1")
os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")

LETTER_TO_INDEX = {"A": 0, "B": 1, "C": 2, "D": 3}


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Unified evaluation for HellaSwag, LAMBADA, and ARC-Easy")
    ap.add_argument("--model", default="/root/code/LLM_from_scratch-assignments/llms-from-scratch-fall-2025-assignment-3/custom_model_hf", help="Path to HF model directory")
    ap.add_argument(
        "--tokenizer",
        default="/thullms/public/Qwen/Qwen3-0.6B-Base",
        help="Tokenizer path (default: /thullms/public/Qwen/Qwen3-0.6B-Base)",
    )
    ap.add_argument("--hellaswag-root", default="/thullms/public/data/Hellaswag", help="HellaSwag data directory")
    ap.add_argument("--lambada-root", default="/thullms/public/data/cimec_lambada", help="LAMBADA data directory")
    ap.add_argument("--arc-file", default="/thullms/public/data/ai2_arc/ARC-Easy/test-00000-of-00001.parquet", help="ARC-Easy parquet file")
    ap.add_argument("--max-samples", type=int, default=None, help="Max samples per benchmark (for quick testing)")
    ap.add_argument("--skip-hellaswag", action="store_true", help="Skip HellaSwag evaluation")
    ap.add_argument("--skip-lambada", action="store_true", help="Skip LAMBADA evaluation")
    ap.add_argument("--skip-arc", action="store_true", help="Skip ARC-Easy evaluation")
    ap.add_argument("--dtype", type=str, default="auto", choices=["auto", "float32", "bfloat16", "float16"], 
                    help="Model dtype (default: auto = bfloat16 on CUDA, float32 on CPU)")
    return ap.parse_args()


# ==================== HellaSwag Functions ====================

def _hellaswag_name_candidates(split: str) -> list[str]:
    split_key = split.lower()
    if split_key in {"validation", "val"}:
        base_names = ["hellaswag_val", "val", "validation"]
    elif split_key in {"train", "training"}:
        base_names = ["hellaswag_train", "train", "training"]
    elif split_key in {"test", "test_seen", "test_unseen"}:
        base_names = ["hellaswag_test", "test"]
    else:
        base_names = [f"hellaswag_{split_key}", split_key]
    patterns = []
    for base in base_names:
        patterns.extend([f"{base}.jsonl", f"{base}.json", f"{base}.jsonl.gz", f"{base}.jsonl.zst", f"{base}.json.gz"])
    return patterns


def _find_hellaswag_file(root: Path, split: str) -> Path | None:
    patterns = _hellaswag_name_candidates(split)
    for pattern in patterns:
        candidate = root / pattern
        if candidate.is_file():
            return candidate
    for pattern in patterns:
        matches = list(root.glob(f"**/{pattern}"))
        if matches:
            return matches[0]
    generic_matches = list(root.glob(f"**/*{split}*.jsonl*"))
    for candidate in generic_matches:
        if candidate.is_file():
            return candidate
    return None


def load_hellaswag(data_root: str, split: str = "validation", max_samples: int | None = None) -> Dataset:
    root = Path(data_root)
    if not root.exists():
        raise FileNotFoundError(f"HellaSwag data directory not found: {root}")
    file_path = _find_hellaswag_file(root, split)
    if file_path is None:
        raise FileNotFoundError(f"Unable to locate HellaSwag split. Expected files like hellaswag_{split}.jsonl under {root}.")
    dataset = load_dataset("json", data_files=str(file_path), split="train")
    if max_samples is not None:
        dataset = dataset.select(range(min(max_samples, len(dataset))))
    return dataset


def _extract_hellaswag_fields(example: dict) -> Tuple[str, list[str], int]:
    context = example.get("ctx")
    if context is None:
        ctx_a = example.get("ctx_a", "")
        ctx_b = example.get("ctx_b", "")
        context = (ctx_a + " " + ctx_b).strip()
    endings = example.get("endings") or example.get("ending_options")
    if not endings:
        raise ValueError("Unable to locate endings in HellaSwag example")
    label = example.get("label")
    if label is None:
        raise ValueError("HellaSwag example missing label")
    if isinstance(label, str):
        label = int(label)
    return context, list(endings), int(label)


def render_hellaswag_example(tokenizer, example: dict) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, int]:
    ctx, endings, label = _extract_hellaswag_fields(example)
    ctx_tokens = tokenizer.encode(ctx, add_special_tokens=False)
    if not isinstance(ctx_tokens, list):
        raise ValueError("Tokenizer.encode must return a list of token ids")
    tok_rows: list[list[int]] = []
    mask_rows: list[list[int]] = []
    for ending in endings:
        end_tokens = tokenizer.encode(" " + ending, add_special_tokens=False)
        if not isinstance(end_tokens, list):
            raise ValueError("Tokenizer.encode must return a list of token ids")
        combined = ctx_tokens + end_tokens
        tok_rows.append(combined)
        mask_rows.append([0] * len(ctx_tokens) + [1] * len(end_tokens))
    max_len = max(len(row) for row in tok_rows)
    pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else (tokenizer.eos_token_id if tokenizer.eos_token_id is not None else 0)
    tokens = torch.full((4, max_len), pad_id, dtype=torch.long)
    attn = torch.zeros((4, max_len), dtype=torch.long)  # Changed from bool to long
    mask = torch.zeros((4, max_len), dtype=torch.long)
    for i, (row, mrow) in enumerate(zip(tok_rows, mask_rows)):
        row_len = len(row)
        tokens[i, :row_len] = torch.tensor(row, dtype=torch.long)
        attn[i, :row_len] = 1  # Changed from True to 1
        mask[i, :len(mrow)] = torch.tensor(mrow, dtype=torch.long)
    return tokens, attn, mask, label


@torch.no_grad()
def evaluate_hellaswag(model, tokenizer, device, dataset) -> dict:
    total = 0
    num_correct = 0
    num_correct_norm = 0
    iterator = tqdm(dataset, total=len(dataset), desc="HellaSwag", dynamic_ncols=True)
    for example in iterator:
        tokens, attn, mask, label = render_hellaswag_example(tokenizer, example)
        tokens = tokens.to(device)
        attn = attn.to(device)
        mask = mask.to(device)
        outputs = model(input_ids=tokens, attention_mask=attn)
        logits = outputs.logits
        shift_logits = logits[..., :-1, :].contiguous()
        shift_tokens = tokens[..., 1:].contiguous()
        shift_attn = attn[..., 1:].contiguous()
        shift_mask = mask[..., 1:].contiguous()
        vocab_size = shift_logits.size(-1)
        losses = F.cross_entropy(shift_logits.view(-1, vocab_size), shift_tokens.view(-1), reduction="none").view(tokens.size(0), -1)
        losses = losses * shift_attn
        comp_mask = shift_mask * shift_attn
        sum_loss = (losses * comp_mask).sum(dim=1)
        token_counts = comp_mask.sum(dim=1).clamp(min=1)
        avg_loss = sum_loss / token_counts
        pred = torch.argmin(sum_loss).item()
        pred_norm = torch.argmin(avg_loss).item()
        num_correct += int(pred == label)
        num_correct_norm += int(pred_norm == label)
        total += 1
    iterator.close()
    return {"num_examples": total, "accuracy": num_correct / max(total, 1), "accuracy_len_norm": num_correct_norm / max(total, 1)}


# ==================== LAMBADA Functions ====================

def _find_parquet_files(root: Path, split: str) -> List[str]:
    matches = []
    if not root.exists():
        return matches
    plain_text = root / "plain_text"
    search_root = plain_text if plain_text.is_dir() else root
    
    # Try multiple patterns
    patterns = [
        f"{split}-*.parquet",
        f"{split}.parquet",
        f"*{split}*.parquet",
    ]
    
    for pattern in patterns:
        found = sorted(str(p) for p in search_root.glob(pattern))
        if found:
            matches.extend(found)
            break
    
    # Also try looking for any .parquet files if nothing found
    if not matches:
        all_parquet = sorted(str(p) for p in search_root.glob("*.parquet"))
        if all_parquet:
            print(f"Warning: Could not find {split}.parquet, trying first available: {all_parquet[0]}")
            matches = all_parquet[:1]
    
    return matches


def load_lambada(data_root: str, split: str = "validation", max_samples: int | None = None) -> Dataset:
    root = Path(data_root)
    if not root.exists():
        raise FileNotFoundError(f"LAMBADA data directory not found: {root}")
    
    files = _find_parquet_files(root, split)
    
    # If parquet files not found, try to find .jsonl or .txt files
    if not files:
        print(f"No parquet files found, searching for alternative formats...")
        jsonl_files = list(root.glob("**/*.jsonl")) + list(root.glob("**/*.json"))
        txt_files = list(root.glob("**/*.txt"))
        
        if jsonl_files:
            print(f"Found JSONL file: {jsonl_files[0]}")
            dataset = load_dataset("json", data_files=str(jsonl_files[0]), split="train")
        elif txt_files:
            print(f"Found TXT file: {txt_files[0]}")
            dataset = load_dataset("text", data_files=str(txt_files[0]), split="train")
        else:
            raise FileNotFoundError(
                f"Unable to locate LAMBADA data files (parquet, jsonl, or txt) within {root}."
            )
    else:
        print(f"Loading LAMBADA from parquet: {files}")
        dataset = load_dataset("parquet", data_files={"data": files}, split="data")
    
    if max_samples is not None:
        dataset = dataset.select(range(min(max_samples, len(dataset))))
    
    return dataset


@torch.no_grad()
def evaluate_lambada(model, tokenizer, device, dataset) -> dict:
    total = 0
    correct = 0
    nll_sum = 0.0
    n_tokens = 0
    progress = tqdm(total=len(dataset), desc="LAMBADA", dynamic_ncols=True)
    for ex in dataset:
        text = ex["text"].strip()
        if not text:
            continue
        parts = text.split()
        if len(parts) < 2:
            continue
        target_word = parts[-1]
        context = " ".join(parts[:-1])
        ctx_ids = tokenizer(context, return_tensors="pt", add_special_tokens=False)["input_ids"].to(device)
        tgt_ids = tokenizer(" " + target_word, return_tensors="pt", add_special_tokens=False)["input_ids"].to(device)
        input_ids = torch.cat([ctx_ids, tgt_ids], dim=1)
        logits = model(input_ids=input_ids).logits
        shift_logits = logits[:, ctx_ids.size(1) - 1:-1, :]
        shift_labels = input_ids[:, ctx_ids.size(1):]
        logprobs = F.log_softmax(shift_logits, dim=-1)
        token_ll = logprobs.gather(-1, shift_labels.unsqueeze(-1)).squeeze(-1)
        nll = -token_ll.sum().item()
        nll_sum += nll
        n_tokens += shift_labels.numel()
        greedy = shift_logits.argmax(dim=-1)
        is_correct = bool(torch.all(greedy == shift_labels).item())
        correct += int(is_correct)
        total += 1
        progress.update(1)
    progress.close()
    return {"num_examples": total, "accuracy": correct / max(1, total), "avg_nll": nll_sum / max(1, n_tokens)}


# ==================== ARC-Easy Functions ====================

def extract_arc_row(row):
    q = row.get("question")
    if isinstance(q, dict):
        question_text = q.get("stem", "")
    else:
        question_text = q or ""
    choices_obj = row.get("choices")
    if isinstance(choices_obj, dict):
        choice_texts = choices_obj.get("text", [])
        choice_labels = choices_obj.get("label", [])
    else:
        choice_texts = row.get("choices.text", [])
        choice_labels = row.get("choices.label", [])
    answer_key = row.get("answerKey")
    return question_text, choice_texts, choice_labels, answer_key


def compute_continuation_loglik(model, tokenizer, context: str, continuation: str, device) -> Tuple[float, int]:
    ctx_ids = tokenizer(context, return_tensors="pt", add_special_tokens=False)["input_ids"].to(device)
    cont_ids = tokenizer(" " + continuation, return_tensors="pt", add_special_tokens=False)["input_ids"].to(device)
    input_ids = torch.cat([ctx_ids, cont_ids], dim=1)
    with torch.no_grad():
        logits = model(input_ids=input_ids).logits
    shift_logits = logits[:, ctx_ids.size(1) - 1:-1, :]
    shift_labels = input_ids[:, ctx_ids.size(1):]
    logprobs = F.log_softmax(shift_logits, dim=-1)
    token_ll = logprobs.gather(-1, shift_labels.unsqueeze(-1)).squeeze(-1)
    total_ll = token_ll.sum().item()
    length = shift_labels.numel()
    return total_ll, length


@torch.no_grad()
def evaluate_arc(model, tokenizer, device, arc_file: str, max_samples: int | None = None) -> dict:
    df = pd.read_parquet(arc_file)
    total = 0
    correct_acc = 0
    correct_norm = 0
    limit = max_samples if max_samples else len(df)
    for _, row in tqdm(df.iterrows(), total=min(len(df), limit), desc="ARC-Easy", dynamic_ncols=True):
        question_text, choice_texts, choice_labels, answer_key = extract_arc_row(row)
        if choice_texts is None or len(choice_texts) == 0 or answer_key is None:
            continue
        gold_idx = None
        if choice_labels is not None and len(choice_labels) > 0 and answer_key in choice_labels:
            labels_list = list(choice_labels)
            gold_idx = labels_list.index(answer_key)
        elif answer_key in LETTER_TO_INDEX:
            gold_idx = LETTER_TO_INDEX[answer_key]
        else:
            continue
        scores = []
        norm_scores = []
        for choice in choice_texts:
            ll, length = compute_continuation_loglik(model, tokenizer, question_text, choice, device)
            scores.append(ll)
            norm_scores.append(ll / max(length, 1))
        pred_idx = int(torch.tensor(scores).argmax().item())
        pred_norm_idx = int(torch.tensor(norm_scores).argmax().item())
        if pred_idx == gold_idx:
            correct_acc += 1
        if pred_norm_idx == gold_idx:
            correct_norm += 1
        total += 1
        if max_samples and total >= max_samples:
            break
    return {"num_examples": total, "accuracy": correct_acc / max(1, total), "accuracy_len_norm": correct_norm / max(1, total)}


# ==================== Main Function ====================

def main() -> None:
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Determine dtype
    if args.dtype == "auto":
        dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32
    elif args.dtype == "float32":
        dtype = torch.float32
    elif args.dtype == "bfloat16":
        dtype = torch.bfloat16
    elif args.dtype == "float16":
        dtype = torch.float16
    else:
        dtype = torch.float32
    
    print(f"Loading model from {args.model}...")
    print(f"Using dtype: {dtype}")
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        args.model, 
        trust_remote_code=True,
        torch_dtype=dtype
    )
    model.to(device)
    model.eval()
    print(f"Model loaded on {device}\n")

    results = {}

    # Evaluate HellaSwag
    if not args.skip_hellaswag:
        try:
            print("=" * 60)
            print("Evaluating HellaSwag...")
            print("=" * 60)
            hellaswag_ds = load_hellaswag(args.hellaswag_root, "validation", args.max_samples)
            results["hellaswag"] = evaluate_hellaswag(model, tokenizer, device, hellaswag_ds)
            print(f"HellaSwag Results: {json.dumps(results['hellaswag'], indent=2)}\n")
        except Exception as e:
            print(f"HellaSwag evaluation failed: {e}\n")
            results["hellaswag"] = {"error": str(e)}

    # Evaluate LAMBADA
    if not args.skip_lambada:
        try:
            print("=" * 60)
            print("Evaluating LAMBADA...")
            print("=" * 60)
            lambada_ds = load_lambada(args.lambada_root, "validation", args.max_samples)
            results["lambada"] = evaluate_lambada(model, tokenizer, device, lambada_ds)
            print(f"LAMBADA Results: {json.dumps(results['lambada'], indent=2)}\n")
        except Exception as e:
            print(f"LAMBADA evaluation failed: {e}\n")
            results["lambada"] = {"error": str(e)}

    # Evaluate ARC-Easy
    if not args.skip_arc:
        try:
            print("=" * 60)
            print("Evaluating ARC-Easy...")
            print("=" * 60)
            results["arc_easy"] = evaluate_arc(model, tokenizer, device, args.arc_file, args.max_samples)
            print(f"ARC-Easy Results: {json.dumps(results['arc_easy'], indent=2)}\n")
        except Exception as e:
            print(f"ARC-Easy evaluation failed: {e}\n")
            results["arc_easy"] = {"error": str(e)}

    # Print final summary
    print("\n" + "=" * 60)
    print("FINAL RESULTS SUMMARY")
    print("=" * 60)
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()