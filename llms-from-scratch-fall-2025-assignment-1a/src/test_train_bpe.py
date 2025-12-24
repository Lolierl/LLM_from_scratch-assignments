import json
import time

from .tokenizer import train_bpe
from .common import FIXTURES_PATH, gpt2_bytes_to_unicode


def test_train_bpe_speed():
    """
    Ensure that BPE training on this small dataset takes no more than 5 seconds. 
    This is a pretty generous upper bound. For reference, my implementation takes 
    0.20 seconds with a few parallelization tricks during pre-tokenization.
    """
    input_path = FIXTURES_PATH / "corpus.en"
    start_time = time.time()
    _, _ = train_bpe(
        corpus=input_path,
        vocab_size=499,
        pattern=r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+""",
    )
    end_time = time.time()
    print(f"train bpe speed: {end_time - start_time}")
    assert end_time - start_time < 5


def test_train_bpe():
    input_path = FIXTURES_PATH / "corpus.en"
    vocab, merges = train_bpe(
        corpus=input_path,
        vocab_size=499,
        pattern=r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+""",
    )

    # Path to the reference tokenizer vocab and merges
    reference_vocab_path = FIXTURES_PATH / "train-bpe-reference-vocab.json"
    reference_merges_path = FIXTURES_PATH / "train-bpe-reference-merges.txt"

    # Compare the learned merges to the expected output merges
    gpt2_byte_decoder = {v: k for k, v in gpt2_bytes_to_unicode().items()}
    with open(reference_merges_path) as f:
        gpt2_reference_merges = [tuple(line.rstrip().split(" ")) for line in f]
        reference_merges = [
            (
                bytes([gpt2_byte_decoder[token] for token in merge_token_1]),
                bytes([gpt2_byte_decoder[token] for token in merge_token_2]),
            )
            for merge_token_1, merge_token_2 in gpt2_reference_merges
        ]
    assert merges == reference_merges

    # Compare the vocab to the expected output vocab
    with open(reference_vocab_path) as f:
        gpt2_reference_vocab = json.load(f)
        reference_vocab = {
            gpt2_vocab_index: bytes([gpt2_byte_decoder[token] for token in gpt2_vocab_item])
            for gpt2_vocab_item, gpt2_vocab_index in gpt2_reference_vocab.items()
        }
    
    # Rather than checking that the vocabs exactly match (since they could
    # have been constructed differently, we'll make sure that the vocab keys and values match)
    
    ## delete the value b'<|endoftext|>' (key 0) from the reference vocab
    for i in range(len(reference_vocab) - 1):
        reference_vocab[i] = reference_vocab[i + 1]
    del reference_vocab[len(reference_vocab) - 1]

    assert set(vocab.keys()) == set(reference_vocab.keys())
    assert set(vocab.values()) == set(reference_vocab.values())

