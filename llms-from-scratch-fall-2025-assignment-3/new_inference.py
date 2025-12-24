"""
SFT Model Testing Script - Tests all required prompts automatically

Usage:
    python chat_with_sft.py \
        --model_path /path/to/sft_hf_model \
        --tokenizer_path /path/to/tokenizer
"""

import argparse
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

# Required prompts from assignment
PROMPTS = [
    "45+33 等于几？",
    "Explain Gradient Descent in 3 sentences",
    "Do you think you are smart?",
    "Which university is better, Tsinghua or Peking?",
    "List 3 challenges Tsinghua students commonly face",
    "Jensen sold 48 GPUs to his friends in April, and then he sold half as many GPUs in May. How many GPUs did Jensen sell altogether in April and May?",
    "Write Python code to check whether a string is a palindrome"
]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', default="/thullms/3022377347/hw3_sft_hf", help='Path to SFT HF model')
    parser.add_argument('--tokenizer_path', default="/thullms/3022377347/Qwen/Qwen3-0.6B-Base", help='Path to tokenizer')
    parser.add_argument('--max_new_tokens', type=int, default=512, help='Max tokens to generate')
    parser.add_argument('--temperature', type=float, default=0.7, help='Sampling temperature')
    parser.add_argument('--top_k', type=int, default=50, help='Top-k sampling')
    parser.add_argument('--repetition_penalty', type=float, default=0.0, help='Repetition penalty (1.0 = no penalty, >1.0 = penalize)')
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu')
    args = parser.parse_args()
    
    print("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_path, trust_remote_code=True)
    
    print("Loading model...")
    model = AutoModelForCausalLM.from_pretrained(
        args.model_path,
        trust_remote_code=True
    ).to(args.device).eval()
    
    print(f"\nModel loaded on {args.device}")
    print("="*80)
    print("Testing SFT Model with Required Prompts")
    print("="*80)
    
    # Test each prompt
    for i, prompt in enumerate(PROMPTS, 1):
        print(f"\n{'='*80}")
        print(f"Prompt {i}/{len(PROMPTS)}: {prompt}")
        print("-"*80)
        
        try:
            # Format as chat
            messages = [{"role": "user", "content": prompt}]
            
            # Apply chat template
            try:
                text = tokenizer.apply_chat_template(
                    messages,
                    tokenize=False,
                    add_generation_prompt=True,
                    enable_thinking=True
                )
            except TypeError:
                text = tokenizer.apply_chat_template(
                    messages,
                    tokenize=False,
                    add_generation_prompt=True
                )
            
            # Tokenize
            inputs = tokenizer(text, return_tensors="pt").to(args.device)
            
            # Generate
            with torch.no_grad():
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=args.max_new_tokens,
                    temperature=args.temperature,
                    top_k=args.top_k,
                    repetition_penalty=args.repetition_penalty,
                    do_sample=True,
                    pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id
                )
            
            # Decode only the generated part
            input_length = inputs.input_ids.shape[1]
            generated_tokens = outputs[0][input_length:]
            response = tokenizer.decode(generated_tokens, skip_special_tokens=True)
            
            print(f"Response:\n{response}")
            
        except Exception as e:
            print(f"Error: {e}")
            continue
    
    print(f"\n{'='*80}")
    print("All prompts tested!")
    print("="*80)


if __name__ == "__main__":
    main()
