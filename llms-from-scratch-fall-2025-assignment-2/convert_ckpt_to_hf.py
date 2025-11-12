"""
Convert your training checkpoint into a local
Hugging Face-compatible folder with config + model weights.

Example:
  python convert_ckpt_to_hf.py \
    --ckpt_path <your model checkpoint path here> \
    --out_dir custom_model_hf

After conversion, you can load the tokenizer and your model with:
  from transformers import AutoTokenizer, AutoModelForCausalLM
  tok = AutoTokenizer.from_pretrained("tokenizer/Qwen3-0.6B-Base", trust_remote_code=True)
  model = AutoModelForCausalLM.from_pretrained("custom_model_hf", trust_remote_code=True)
"""

import argparse
import os
import shutil
import torch

from custom_model.configuration_custom_model import CustomModelConfig
from custom_model.modeling_custom_model import CustomModelForCausalLM


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt_path", required=True, help="Path to .pt checkpoint from training")
    ap.add_argument("--out_dir", required=True, help="Output folder for HF model")
    return ap.parse_args()


def main():
    args = parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    print(f"[info] Loading training checkpoint from {args.ckpt_path}")
    ckpt = torch.load(args.ckpt_path, map_location="cpu")
    model_args = ckpt["model_args"]
    if model_args is None:
        print("[warn] model_args not found in checkpoint, using manual configuration.")
        model_args = {
            "vocab_size": 151936,
            "embed_dim": 512,
            "inter_dim": 1024,
            "num_attention_heads": 8,
            "num_key_value_heads": 4,
            "head_dim": 64,
            "n_layers": 12,
            "rope_theta": 1_000_000.0,
        }

    cfg = CustomModelConfig(**model_args)

    CustomModelConfig.register_for_auto_class()
    CustomModelForCausalLM.register_for_auto_class("AutoModelForCausalLM")

    print("[info] Instantiating HF model wrapper and loading weights...")
    model = CustomModelForCausalLM(cfg)
    missing, unexpected = model.transformer.load_state_dict(ckpt["model"], strict=False)
    if missing:
        print("[warn] Missing keys:", missing)
    if unexpected:
        print("[warn] Unexpected keys:", unexpected)

    print(f"[info] Saving HF model to {args.out_dir}")
    model.save_pretrained(args.out_dir, safe_serialization=False)

    cfg.save_pretrained(args.out_dir)

    # Make the exported folder self-contained by copying the code package
    pkg_src = os.path.join(os.path.dirname(__file__), "custom_model")
    pkg_dst = os.path.join(args.out_dir, "custom_model")
    if not os.path.exists(pkg_dst):
        print(f"[info] Copying model code to {pkg_dst}")
        shutil.copytree(pkg_src, pkg_dst)

    print("[done] Export complete. You can now load with AutoModelForCausalLM.from_pretrained(..., trust_remote_code=True)")


if __name__ == "__main__":
    main()
