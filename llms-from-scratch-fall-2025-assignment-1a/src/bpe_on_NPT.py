from common import FIXTURES_PATH, gpt2_bytes_to_unicode
from tokenizer import train_bpe
import json
from typing import Dict, List, Tuple

input_path = FIXTURES_PATH / "text" / "NPT-en.txt"


def save_vocab(vocab: Dict[int, bytes], vocab_path: str):
    """Saves the vocabulary to a JSON file in the GPT-2 format."""
    print(f"Saving vocabulary to {vocab_path}...")
    # We need to swap keys and values and decode bytes to printable strings
    byte_encoder = gpt2_bytes_to_unicode()
    json_vocab = {
        "".join(byte_encoder[b] for b in token_bytes): token_id
        for token_id, token_bytes in vocab.items()
    }

    with open(vocab_path, "w", encoding="utf-8") as f:
        json.dump(json_vocab, f, ensure_ascii=False)
    print("Done.")


def save_merges(merges: List[Tuple[bytes, bytes]], merges_path: str):
    print(f"Saving merges to {merges_path}...")
    byte_encoder = gpt2_bytes_to_unicode()
    with open(merges_path, "w", encoding="utf-8") as f:
        for p1, p2 in merges:
            s1 = "".join(byte_encoder[b] for b in p1)
            s2 = "".join(byte_encoder[b] for b in p2)
            f.write(f"{s1} {s2}\n")
    print("Done.")


def main():
    #-------------------- P1.6b: YOUR CODE HERE ----------------------
    vocab_size = 2000
    pattern = r"(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}" \
    r"| ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+"
    #-----------------------------------------------------------------
    vocab, merges = train_bpe(input_path, vocab_size, pattern)
    output_vocab_path = "NPT_vocab.json"
    output_merges_path = "NPT_merges.txt"

    save_vocab(vocab, output_vocab_path)
    save_merges(merges, output_merges_path)


if __name__ == "__main__":
    main()
