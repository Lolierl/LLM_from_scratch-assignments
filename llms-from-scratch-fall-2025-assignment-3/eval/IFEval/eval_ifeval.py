""" 
You can run the evaluation with the following command:
python eval/IFEval/eval_ifeval.py \
  --input_data <path to the IFEval jsonl file> \
  --model_path <path to your hf-converted model> \
  --tokenizer_path <path to the tokenizer> \
  --output_json <path to save the json results> \
  --device cuda:0
"""
import argparse
import json
import os
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from tqdm import tqdm
import instructions_registry as instructions_registry


def load_ifeval_data(path: str):
    """
    Load IFEval data from jsonl file.
    """
    data = []
    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                data.append(json.loads(line))
    return data

def _preprocess_response(response: str):
    """
    Preprocess response for loose evaluation.
    """
    r = response.split("\n")
    response_remove_first = "\n".join(r[1:]).strip()
    response_remove_last = "\n".join(r[:-1]).strip()
    response_remove_both = "\n".join(r[1:-1]).strip()
    revised_response = response.replace("*", "")
    revised_response_remove_first = response_remove_first.replace("*", "")
    revised_response_remove_last = response_remove_last.replace("*", "")
    revised_response_remove_both = response_remove_both.replace("*", "")
    all_responses = [
        response,
        revised_response,
        response_remove_first,
        response_remove_last,
        response_remove_both,
        revised_response_remove_first,
        revised_response_remove_last,
        revised_response_remove_both,
    ]
    return all_responses

def evaluate_sample(response, prompt, instruction_list, all_kwargs):
    """
    Evaluate a single sample against its instructions.
    Returns (strict_pass, loose_pass, strict_details, loose_details)
    """
    if instructions_registry is None:
        return False, False, [], []

    # Loose instructions preprocessing
    all_responses = _preprocess_response(response)

    is_following_list_strict = []
    is_following_list_loose = []

    for index, instruction_id in enumerate(instruction_list):
        instruction_cls = instructions_registry.INSTRUCTION_DICT[instruction_id]
        instruction = instruction_cls(instruction_id)

        # Remove None values from kwargs
        task_kwargs = {k: v for k, v in all_kwargs[index].items() if v}
        instruction.build_description(**task_kwargs)
        args = instruction.get_instruction_args()
        if args and "prompt" in args:
            instruction.build_description(prompt=prompt)

        # Strict
        if response.strip() and instruction.check_following(response):
            is_following_list_strict.append(True)
        else:
            is_following_list_strict.append(False)

        # Loose
        is_following = False
        for r in all_responses:
            if r.strip() and instruction.check_following(r):
                is_following = True
                break
        is_following_list_loose.append(is_following)

    return (
        all(is_following_list_strict),
        all(is_following_list_loose),
        is_following_list_strict,
        is_following_list_loose
    )

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_data', default='/thullms/public/data/IFEval/ifeval_input_data.jsonl', help='Path to IFEval jsonl file')
    parser.add_argument('--model_path', required=True, help='Path to model checkpoint or HF repo')
    parser.add_argument('--tokenizer_path', type=str, default="/thullms/3022377347/Qwen/Qwen3-0.6B-Base", help='Path to tokenizer')
    parser.add_argument('--max_samples', type=int, default=-1, help='Limit samples for testing')
    parser.add_argument('--output_json', type=str, default=None, help='Path to save results')
    parser.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    args = parser.parse_args()

    print(f"Loading tokenizer from {args.tokenizer_path} ...")
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_path, trust_remote_code=True)

    print(f"Loading model from {args.model_path} ...")
    model = AutoModelForCausalLM.from_pretrained(args.model_path, trust_remote_code=True).eval()
    model.to(args.device)

    print(f"Reading data from {args.input_data} ...")
    data = load_ifeval_data(args.input_data)
    
    if args.max_samples > 0:
        data = data[:args.max_samples]

    print(f"Evaluating {len(data)} samples...")
    
    results = []
    strict_correct_count = 0
    loose_correct_count = 0
    total = 0

    for item in tqdm(data):
        prompt = item['prompt']
        instruction_list = item['instruction_id_list']
        kwargs = item['kwargs']
        
        # TODO: Implement the generation logic here.
        # Steps:
        # 1. Create a list of messages containing the user prompt.
        #    Example: messages = [{"role": "user", "content": prompt}]
        # 2. Apply the chat template to prepare inputs for the model.
        #    Use tokenizer.apply_chat_template()
        # 3. Generate the output using model.generate().
        # 4. Decode the generated tokens to get the final response string.
        #    Make sure to skip special tokens and remove the input prompt from the output.
        messages = [{"role": "user", "content": prompt}]
        inputs = tokenizer.apply_chat_template(messages, return_dict=True, return_tensors="pt").to(args.device)
        with torch.no_grad():
            generated_ids = model.generate(**inputs, max_new_tokens=1280, temperature=0.7, top_k=50)
        generated_ids = generated_ids[:, inputs['input_ids'].shape[1]:]
        response = tokenizer.decode(generated_ids[0], skip_special_tokens=True)
        
        # Evaluate
        strict_pass, loose_pass, strict_details, loose_details = evaluate_sample(
            response, prompt, instruction_list, kwargs
        )
        
        if strict_pass:
            strict_correct_count += 1
        if loose_pass:
            loose_correct_count += 1
        total += 1
        
        results.append({
            "key": item.get("key"),
            "prompt": prompt,
            "response": response,
            "instruction_id_list": instruction_list,
            "kwargs": kwargs,
            "strict_pass": strict_pass,
            "loose_pass": loose_pass,
            "strict_details": strict_details,
            "loose_details": loose_details
        })

    strict_acc = strict_correct_count / total if total > 0 else 0.0
    loose_acc = loose_correct_count / total if total > 0 else 0.0
    
    print(f"\nStrict Accuracy: {strict_acc:.4f} ({strict_correct_count}/{total})")
    print(f"Loose Accuracy: {loose_acc:.4f} ({loose_correct_count}/{total})")

    if args.output_json:
        os.makedirs(os.path.dirname(args.output_json) or '.', exist_ok=True)
        with open(args.output_json, 'w', encoding='utf-8') as f:
            json.dump({
                "strict_accuracy": strict_acc, 
                "loose_accuracy": loose_acc, 
                "samples": results
            }, f, indent=2, ensure_ascii=False)
        print(f"Saved results to {args.output_json}")

if __name__ == "__main__":
    main()
