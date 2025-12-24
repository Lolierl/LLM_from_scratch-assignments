import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

# Point to your SFT-exported HF folder (convert_ckpt_to_hf.py output)
MODEL_PATH = "/thullms/3022377347/hw3_sft_hf_final"
TOKENIZER_PATH = "/thullms/3022377347/Qwen/Qwen3-0.6B-Base"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

prompts = [
    "45+33 等于几？",
    "Explain Gradient Descent in 3 sentences.",
    "Do you think you are smart?",
    "Which university is better, Tsinghua or Peking?",
    "List 3 challenges Tsinghua students commonly face.",
    "Jensen sold 48 GPUs to his friends in April, and then he sold half as many GPUs in May. "
    "How many GPUs did Jensen sell altogether in April and May?",
    "Write Python code to check whether a string is a palindrome."
]

def main():
    tokenizer = AutoTokenizer.from_pretrained(
        TOKENIZER_PATH,
        trust_remote_code=True, 
        enable_thinking=True
    )
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_PATH,
        trust_remote_code=True
    ).to(DEVICE).eval()

    for i, prompt in enumerate(prompts, 1):
        messages = [{"role": "user", "content": prompt}]

        inputs = tokenizer.apply_chat_template(
                messages,
                return_dict=True,
                return_tensors="pt", 
                add_generation_prompt=True, 
                add_special_tokens=False
            )

        inputs = {k: v.to(DEVICE) for k, v in inputs.items()}

        with torch.no_grad():
            # For debugging stability, try greedy first (no sampling).
            output_ids = model.generate(
                **inputs,
                max_new_tokens=512,
                do_sample=True,
                temperature=0.7,
                top_k=50,
                eos_token_id=tokenizer.eos_token_id,
            )

        output_ids = output_ids[:, inputs["input_ids"].shape[1]:]
        response = tokenizer.decode(
            output_ids[0],
            skip_special_tokens=True
        )
        print("=" * 60)
        print(f"[Prompt {i}] {prompt}")
        print("-" * 60)
        print(response.strip())
        print()

if __name__ == "__main__":
    main()
