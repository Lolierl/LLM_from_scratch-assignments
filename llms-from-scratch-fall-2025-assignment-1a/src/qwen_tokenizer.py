import os
import json
from .tokenizer import BPETokenizer
from functools import lru_cache
from .common import FIXTURES_PATH

@lru_cache
def gpt2_bytes_to_unicode() -> dict[int, str]:
    """
    Returns a mapping between every possible byte (an integer from 0 to 255) to a
    printable unicode string character representation. This function is taken
    from the GPT-2 code.

    For example, `chr(0)` is `\x00`, which is an unprintable character:

    >>> chr(0)
    '\x00'
    >>> print(chr(0))

    As a result, this function returns a dictionary `d` where `d[0]` returns `Ā`.
    The bytes that are visually printable keep their original string representation [1].
    For example, `chr(33)` returns `!`, and so accordingly `d[33]` returns `!`.
    Note in particular that the space character `chr(32)` becomes `d[32]`, which
    returns 'Ġ'.

    For unprintable characters, the function shifts takes the integer representing
    the Unicode code point of that character (returned by the Python `ord`) function
    and shifts it by 256. For example, `ord(" ")` returns `32`, so the the space character
    ' ' is shifted to `256 + 32`. Since `chr(256 + 32)` returns `Ġ`, we use that as the
    string representation of the space.

    This function can simplify the BPE implementation and makes it slightly easier to
    manually inspect the generated merges after they're serialized to a file.
    """
    # These 188 integers can used as-is, since they are not whitespace or control characters.
    # See https://www.ssec.wisc.edu/~tomw/java/unicode.html.
    bs = list(range(ord("!"), ord("~") + 1)) + list(range(ord("¡"), ord("¬") + 1)) + list(range(ord("®"), ord("ÿ") + 1))
    cs = bs[:]
    # now get the representations of the other 68 integers that do need shifting
    # each will get mapped chr(256 + n), where n will grow from 0...67 in the loop
    # Get printable representations of the remaining integers 68 integers.
    n = 0
    for b in range(2**8):
        if b not in bs:
            # If this integer isn't in our list of visually-representable
            # charcters, then map it to the next nice character (offset by 256)
            bs.append(b)
            cs.append(2**8 + n)
            n += 1
    characters = [chr(n) for n in cs]
    d = dict(zip(bs, characters))
    return d


def load_qwen_vocab_and_merges():
    vocab_path = FIXTURES_PATH / 'qwen3_tokenizer' / 'vocab.json'
    merges_path = FIXTURES_PATH / 'qwen3_tokenizer' / 'merges.txt'

    gpt2_byte_decoder = {v: k for k, v in gpt2_bytes_to_unicode().items()}
    with open(vocab_path) as vocab_f:
        gpt2_vocab = json.load(vocab_f)
    gpt2_bpe_merges = []
    with open(merges_path) as f:
        for line in f:
            if line.startswith("#version"):
                continue
            cleaned_line = line.rstrip()
            if cleaned_line and len(cleaned_line.split(" ")) == 2:
                gpt2_bpe_merges.append(tuple(cleaned_line.split(" ")))
    vocab = {
        gpt2_vocab_index: bytes([gpt2_byte_decoder[token] for token in gpt2_vocab_item])
        for gpt2_vocab_item, gpt2_vocab_index in gpt2_vocab.items()
    }

    merges = [
        (
            bytes([gpt2_byte_decoder[token] for token in merge_token_1]),
            bytes([gpt2_byte_decoder[token] for token in merge_token_2]),
        )
        for merge_token_1, merge_token_2 in gpt2_bpe_merges
    ]
    return vocab, merges


def load_qwen_tokenizer_from_dir() -> BPETokenizer:
    vocab, merges = load_qwen_vocab_and_merges()
    
    #-------------------- P1.5: YOUR CODE HERE ----------------------
    pattern = r"(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}" \
    r"| ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+"
    tokenizer = BPETokenizer(pattern, vocab, merges)
    return tokenizer
    #----------------------------------------------------------------


def encode_file_with_qwen_tokenizer(
    input_path: str,
) -> list[int]:
    """
    Reads text from an input file, encodes it using the qwen tokenizer, and 
    returns the list of token IDs.
    """
    tokenizer = load_qwen_tokenizer_from_dir()

    #-------------------- P1.5: YOUR CODE HERE ----------------------
    with open(input_path, 'r', encoding='utf-8') as f:
        text = f.read()
    token_ids = tokenizer.encode(text)
    return token_ids
    #----------------------------------------------------------------


def decode_sequence_with_qwen_tokenizer(
    token_ids: list[int],
) -> str:
    """
    Decodes a list of token IDs using the qwen tokenizer and returns the decoded text.
    """
    tokenizer = load_qwen_tokenizer_from_dir()

    #-------------------- P1.5: YOUR CODE HERE ----------------------
    text = tokenizer.decode(token_ids)
    return text
    #----------------------------------------------------------------

def calculate():
    input_path = FIXTURES_PATH / "text" / "NPT-fr.txt"
    with open(input_path, 'r', encoding='utf-8') as f:
        text: str = f.read()
    encoded_bytes = text.encode('utf-8')
    print(f"Input text has {len(text)} characters and {len(encoded_bytes)} bytes.")
    token_ids = encode_file_with_qwen_tokenizer(input_path)
    print(f"Encoded {len(token_ids)} tokens.")
    avg_token_length = len(encoded_bytes) / len(token_ids)
    print(f"Average token length: {avg_token_length:.2f} bytes.")

if __name__ == "__main__":
    import matplotlib.pyplot as plt

    tokenizer = load_qwen_tokenizer_from_dir()

    ns = list(range(1, 31))

    def token_counts_for_texts():
        a_counts = []
        b_counts = []
        c_counts = []
        d_counts = []
        for n in ns:
            a = "哈" * n
            b = ("不容易" * 1000)[:n]
            c = "5" * n
            d = "\n" * n

            a_counts.append(len(tokenizer.encode(a)))
            b_counts.append(len(tokenizer.encode(b)))
            c_counts.append(len(tokenizer.encode(c)))
            d_counts.append(len(tokenizer.encode(d)))
        return a_counts, b_counts, c_counts, d_counts

    a_counts, b_counts, c_counts, d_counts = token_counts_for_texts()

    def plot_counts(ns, counts, title, fname):
        plt.figure()
        plt.plot(ns, counts, marker="o")
        plt.xlabel("n")
        plt.ylabel("Number of tokens")
        plt.title(title)
        plt.xticks(ns)
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(fname)
        plt.show()

    plot_counts(ns, a_counts, "Tokens for '哈' * n", "tokens_hash_n.png")
    plot_counts(ns, b_counts, "Tokens for ('不容易' * 1000)[:n]", "tokens_bu_rong_yi_prefix_n.png")
    plot_counts(ns, c_counts, "Tokens for '5' * n", "tokens_digit5_n.png")
    plot_counts(ns, d_counts, "Tokens for '\\n' * n", "tokens_newline_n.png")