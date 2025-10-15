"""
Interactive command-line chat application for a local Qwen3 model.

This script loads a pre-trained model and tokenizer and provides a terminal
interface for users to interact with the model. Generation parameters can be
customized via command-line arguments.

Usage:
    python chat.py --top_k 50 --temperature 0.7 --max_new_tokens 512
"""
import json
import os
import urllib.request
import argparse
from pathlib import Path
import torch
from safetensors.torch import load_file
from transformers import AutoTokenizer
from src.model import ModelArgs, Transformer

BASE_PATH = Path("src") / "fixtures"
MODEL_WEIGHTS_PATH = BASE_PATH / "qwen3_0.6b_weights" / "qwen3-0.6B.safetensors"
TOKENIZER_PATH = BASE_PATH / "qwen3_tokenizer"
MODEL_ARCH_PATH = BASE_PATH / "architectures" / "qwen3-0.6B.json"
WEIGHTS_DOWNLOAD_URL = 'https://cloud.tsinghua.edu.cn/f/9ae7ca9806254b42bf4a/?dl=1'

def load_resources():
    print("Initializing model ...")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"--> Using device: {device}")

    tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_PATH)
    with open(MODEL_ARCH_PATH, 'r') as f:
        model_args = ModelArgs(**json.load(f))
    model = Transformer(model_args)

    if not MODEL_WEIGHTS_PATH.exists():
        MODEL_WEIGHTS_PATH.parent.mkdir(parents=True, exist_ok=True)
        urllib.request.urlretrieve(WEIGHTS_DOWNLOAD_URL, MODEL_WEIGHTS_PATH)

    model.load_state_dict(load_file(MODEL_WEIGHTS_PATH))    
    model.to(device)
    model.eval()

    print("Model loaded successfully!")
    return model, tokenizer, device

def generate_response(model, tokenizer, device, prompt, args):
    message = [{"role": "user", "content": prompt}]
    
    input_batch = tokenizer.apply_chat_template(
        message,
        add_generation_prompt=True,
        tokenize=True,
        return_dict=True,
        return_tensors="pt",
        enable_thinking=True
    ).to(device)

    try:
        output_batch = model.generate(
            **input_batch,
            max_new_tokens=args.max_new_tokens,
            top_k=args.top_k,
            temperature=args.temperature,
            eos_token_id=tokenizer.eos_token_id
        )
        
        input_token_len = input_batch["input_ids"].shape[1]
        output_str = tokenizer.decode(output_batch[0][input_token_len:], skip_special_tokens=True)
        return output_str

    except Exception as e:
        return f"\nError during generation: {e}"

def main():
    parser = argparse.ArgumentParser(description="Interactive chat with a local Qwen3 model.")
    parser.add_argument('--top_k', '-k', type=int, default=50)
    parser.add_argument('--temperature', '-t', type=float, default=0.7)
    parser.add_argument('--max_new_tokens', '-m', type=int, default=64)
    args = parser.parse_args()

    if args.top_k <= 0:
        print("! Error: --top_k must be a positive integer.")
        exit(1)
    if args.temperature < 0.0:
        print("! Error: --temperature must be a non-negative float.")
        exit(1)
    if args.max_new_tokens <= 0 and args.max_new_tokens != -1:
        print("! Error: --max_new_tokens must be a positive integer or -1.")
        exit(1)

    model, tokenizer, device = load_resources()

    print("\n" + "="*50)
    print("Chat session started. Type 'quit' or 'exit' to end.")
    print(f"   Parameters: top_k={args.top_k}, temperature={args.temperature}, max_new_tokens={args.max_new_tokens}")
    print("="*50 + "\n")

    try:
        while True:
            prompt = input("You: ")
            if prompt.lower() in ["quit", "exit"]:
                print("Chat ended. Goodbye!")
                break
            
            print("Model: ", end="", flush=True)
            response = generate_response(model, tokenizer, device, prompt, args)
            print(response)

    except KeyboardInterrupt:
        print("\nExiting due to user interruption. Goodbye!")

if __name__ == "__main__":
    main()
