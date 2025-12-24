from transformers import AutoTokenizer
from .model import *
import json
import torch
from safetensors.torch import load_file
from .common import FIXTURES_PATH
import urllib.request
import os
import time

local_model_path = FIXTURES_PATH / "qwen3_0.6b_weights" / "qwen3-0.6B.safetensors"
local_tokenizer_path = FIXTURES_PATH / "qwen3_tokenizer"
tokenizer = None
model = None

# test cases

prompt_batch = [
    "Tell me how to put an elephant in a refrigerator.",
    "What is the capital of Japan?",
    "Describe the color blue to someone who has never seen it.",
    "Give me a simple recipe for making scrambled eggs.",
    "Write a Python function to calculate the factorial of a number."
]

expected_len1000 = """<think>\nOkay, the user is asking how to put an elephant in a refrigerator. Hmm, that\'s a bit of a trick question. Let me think. First, I need to consider the context. The user might be testing if I can come up with a creative or humorous answer. But wait, elephants are big animals, and refrigerators are designed for cold storage. Putting an elephant inside would be impossible because of the size and the temperature. \n\nI should explain that putting an elephant in a refrigerator is not feasible. Maybe mention the physical limitations. Also, perhaps add a playful twist, like a joke about the elephant\'s size. But I need to make sure the answer is clear and helpful. Let me check if there\'s any other angle. Maybe the user is looking for a pun or a play on words. But I think the straightforward answer is best here. Let me structure the response to explain the impossibility and maybe offer a humorous take.\n</think>\n\nPutting an elephant in a refrigerator is not possible because elephants are much larger than refrigerators, and refrigerators are designed to maintain a cold environment. Here\'s why:  \n\n1. **Physical Limitations**: Refrigerators are made of metal and are designed to hold objects in a controlled temperature. An elephant’s size would require a massive space, which is impossible.  \n2. **Temperature Constraints**: Refrigerators operate at temperatures below freezing, but elephants can’t survive in such a cold environment.  \n3. **Humorous Twist**: If you’re looking for a playful answer, you could say, *"An elephant in a refrigerator? That’s a classic joke—just don’t forget to bring a blanket!"*  \n\nIn short, the answer is simply: **Impossible.**"""     

expected_len100 = [
    """<think>\nOkay, the user is asking how to put an elephant in a refrigerator. Hmm, that's a bit of a trick question. Let me think. First, I need to consider the context. The user might be testing if I can come up with a creative or humorous answer. But wait, elephants are big animals, and refrigerators are designed for cold storage. Putting an elephant inside would be impossible because of the size and the temperature. \n\nI should explain that putting an elephant in a""",
    """<think>\nOkay, the user is asking about the capital of Japan. I need to make sure I recall the correct answer. Japan's capital is Tokyo. Let me think... Yes, Tokyo is the capital city. I should confirm that there isn't any other city that's considered the capital. Also, maybe mention that it's a major city in the country. I should keep the response straightforward and accurate.\n</think>\n\nThe capital of Japan is **Tokyo**.""",
    """<think>\nOkay, the user wants me to describe blue to someone who hasn't seen it. Let me start by thinking about what blue is. It's a color that's associated with the sky, oceans, and the sky blue. But how to explain it without them seeing it?\n\nMaybe start with the sky. Blue is the sky's color, so I can mention how it's like a deep blue. Then talk about the ocean, which is also blue. Maybe compare it to the sky.""",
    """<think>\nOkay, the user wants a simple recipe for making scrambled eggs. Let me start by recalling the basic ingredients. They usually need eggs, milk, salt, pepper, maybe some oil for frying. But since they asked for a simple recipe, I should avoid complicated steps.\n\nFirst, I should mention the main components: eggs, milk, salt, pepper, and oil. Maybe suggest using a bowl or a pan. Then, the process: whisking the eggs, adding the milk, and""",
    """<think>\nOkay, I need to write a Python function to calculate the factorial of a number. Let me think about how to approach this.\n\nFirst, what's a factorial? Oh right, the product of all positive integers up to that number. So 5! is 5*4*3*2*1 = 120. But wait, what if the number is zero or negative? Oh right, the factorial of zero is 1, and negative numbers aren't defined in"""
]

# prepare model and tokenizer

def load_tokenizer_and_model():
    global tokenizer, model
    if tokenizer is not None and model is not None:
        return
    tokenizer = AutoTokenizer.from_pretrained(local_tokenizer_path)

    with open(FIXTURES_PATH / "architectures/qwen3-0.6B.json", 'r') as f:
        model_args = ModelArgs(**json.load(f))
    model = Transformer(model_args)
    if not os.path.exists(local_model_path):
        os.makedirs(local_model_path.parent, exist_ok=True)
        try:
            urllib.request.urlretrieve('https://cloud.tsinghua.edu.cn/f/9ae7ca9806254b42bf4a/?dl=1', local_model_path)
        except Exception as e:
            print(f"Error downloading model weights: {e}")
            raise Exception(f"!!!ATTENTION!!! Please download the model weights manually with the URL 'https://cloud.tsinghua.edu.cn/f/9ae7ca9806254b42bf4a/?dl=1' and place the downloaded file at {local_model_path}. The script itself failed to download these weights.")
    model.load_state_dict(load_file(local_model_path))


def test_single_generation():
    load_tokenizer_and_model()
    message = [{"role": "user", "content": prompt_batch[0]}]
    text = tokenizer.apply_chat_template(
        message,
        add_generation_prompt=True,
        tokenize=False,
        enable_thinking=True
    )   
    print("--- Prompt ---")
    print(text)
    print("-" * 30)

    input_batch = tokenizer.apply_chat_template(
        message,
        add_generation_prompt=True,
        tokenize=True,
        padding=True,  # Enable padding to the longest sequence
        return_dict=True,
        return_tensors="pt",
        enable_thinking=True,
    )

    print(input_batch)

    output_batch = model.generate(**input_batch, max_new_tokens=1000, top_k=1, eos_token_id=tokenizer.eos_token_id)
    input_token_len = input_batch["input_ids"].shape[1]
    output_str = tokenizer.decode(output_batch[0][input_token_len:], skip_special_tokens=True)

    print("--- Generated Response ---")
    print(output_str)
    print("-" * 30)

    assert output_str == expected_len1000


def test_batch_generation():
    load_tokenizer_and_model()
    message_batch = [
        [{"role": "user", "content": p}] for p in prompt_batch
    ]

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    text_batch = [tokenizer.apply_chat_template(
        message,
        add_generation_prompt=True,
        tokenize=False,
        enable_thinking=True
    ) for message in message_batch]

    print("--- Prompts ---")
    for tid, text in enumerate(text_batch):
        print(f"Prompt {tid}:\n{repr(text)}\n")
        print("-" * 30)

    input_batch = tokenizer.apply_chat_template(
        message_batch,
        add_generation_prompt=True,
        tokenize=True,
        padding=True,  # Enable padding to the longest sequence
        return_dict=True,
        return_tensors="pt",
        enable_thinking=True,
    )

    print(input_batch)

    output_batch = model.generate(**input_batch, max_new_tokens=100, top_k=1, eos_token_id=tokenizer.eos_token_id)
    output_strs = []

    print("--- Generated Responses ---")
    input_token_len = input_batch["input_ids"].shape[1]

    for i, output_tokens in enumerate(output_batch):
        # Slice the output tokens to get only the generated part
        generated_tokens = output_tokens[input_token_len:]
        output_str = tokenizer.decode(generated_tokens, skip_special_tokens=True)
        
        print(f"Response {i}:\n{repr(output_str)}\n")
        print("-" * 30)
        output_strs.append(output_str)

    assert output_strs == expected_len100


def test_kvcache():
    load_tokenizer_and_model()
    model.to("cpu")
    start_time = time.time()
    test_batch_generation()
    end_time = time.time()
    assert end_time - start_time <= 120


