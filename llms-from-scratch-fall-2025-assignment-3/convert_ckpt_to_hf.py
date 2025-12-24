"""
Convert your training checkpoint into a local
Hugging Face-compatible folder with config + model weights.

Example:
  python convert_ckpt_to_hf.py \
    --ckpt_path /thullms/3022377347/sft_fixed2_checkpoints/checkpoint_final.pt \
    --out_dir /thullms/3022377347/hw3_sft_hf

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
    model_args = ckpt.get("model_args", None)
    if model_args is None:
        print("[warn] model_args not found, loading from qwen3-0.6B.json")
        import json
        with open("qwen3-0.6B.json", "r") as f:
            model_args = json.load(f)
    print(model_args)   
    cfg = CustomModelConfig(**model_args)

    CustomModelConfig.register_for_auto_class()
    CustomModelForCausalLM.register_for_auto_class("AutoModelForCausalLM")

    print("[info] Instantiating HF model wrapper and loading weights...")
    model = CustomModelForCausalLM(cfg)
    state_dict = ckpt["model_state_dict"]
    
    fixed_state_dict = {}
    for k, v in state_dict.items():
        k = k.replace("_orig_mod.", "")
        fixed_state_dict[k] = v
    ckpt["model_state_dict"] = fixed_state_dict
    missing, unexpected = model.transformer.load_state_dict(ckpt["model_state_dict"], strict=False)
    
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
