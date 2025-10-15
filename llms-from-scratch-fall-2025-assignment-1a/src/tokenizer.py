from __future__ import annotations
import regex
#-------------------- P1.4: YOUR CODE HERE ----------------------
class BPETokenizer:
    def __init__(self, pattern: str, vocab: dict[int, bytes], merges: list[tuple[bytes, bytes]]):
        self.pattern = pattern
        self.vocab = vocab
        self.merges = merges
        self.token_to_id = {v: k for k, v in vocab.items()}
    
    def encode(self, text: str) -> list[int]:
        pre_tokens = regex.findall(self.pattern, text)
        tokens = []
        for pre_token in pre_tokens:
            encoding = [bytes([b]) for b in pre_token.encode('utf-8')]

            for rule in self.merges:
                i = 0
                while i < len(encoding) - 1:
                    if (encoding[i], encoding[i + 1]) == rule:
                        encoding[i] = encoding[i] + encoding[i + 1]
                        del encoding[i + 1]
                    else:
                        i += 1
            tokens.extend(encoding)

        token_ids = []
        for token in tokens:
            token_ids.append(self.token_to_id[token])
        return token_ids
    
    def decode(self, encoding: list[int]) -> str:
        tokens = [self.vocab[id] for id in encoding]
        text = b''.join(tokens).decode('utf-8', errors='ignore')
        return text
#----------------------------------------------------------------

            
def train_bpe(
    corpus: str,
    vocab_size: int,
    pattern: str,
) -> tuple[dict[int, bytes], list[tuple[bytes, bytes]]]:
    """Given the path to an input corpus, run train a BPE tokenizer and
    output its vocabulary and merges.

    Args:
        corpus (str): Path to BPE tokenizer training data.
        vocab_size (int): Total number of items in the tokenizer's vocabulary.
        pattern (str): A regex pattern to pre-tokenize the input text.

    Returns:
        tuple[dict[int, bytes], list[tuple[bytes, bytes]]]:
            vocab:
                The trained tokenizer vocabulary, a mapping from int (token ID in the vocabulary)
                to bytes (token bytes)
            merges:
                BPE merges. Each list item is a tuple of bytes (<token1>, <token2>),
                representing that <token1> was merged with <token2>.
                Merges are ordered by order of creation.
    """

    #-------------------- P1.6a: YOUR CODE HERE ----------------------
    from collections import Counter

    with open(corpus, 'r', encoding='utf-8') as f:
        text = f.read()
    pre_tokens = regex.findall(pattern, text)

    vocab = {i: bytes((i,)) for i in range(256)}
    current_vocab_size = 256

    merges: list[tuple[bytes, bytes]] = []

    counter: Counter = Counter()
    for pre_token in pre_tokens:
        key = tuple(bytes((b,)) for b in pre_token.encode('utf-8'))
        counter[key] += 1

    while current_vocab_size < vocab_size:
        pair_counts: Counter = Counter()
        for seq, seq_count in counter.items():
            if len(seq) < 2:
                continue
            for i in range(len(seq) - 1):
                pair_counts[(seq[i], seq[i + 1])] += seq_count

        if not pair_counts:
            break

        most_frequent_pair, best_count = max(pair_counts.items(), key=lambda x: (x[1], x[0]))
        if best_count < 2:
            break

        merged = most_frequent_pair[0] + most_frequent_pair[1]
        vocab[current_vocab_size] = merged
        merges.append(most_frequent_pair)
        current_vocab_size += 1

        new_counter: Counter = Counter()
        for seq, seq_count in counter.items():
            if not seq:
                new_counter[seq] += seq_count
                continue

            new_seq: list[bytes] = []
            i = 0
            while i < len(seq):
                if i + 1 < len(seq) and (seq[i], seq[i + 1]) == most_frequent_pair:
                    new_seq.append(merged)
                    i += 2
                else:
                    new_seq.append(seq[i])
                    i += 1
            new_counter[tuple(new_seq)] += seq_count

        counter = new_counter

    return vocab, merges
    #-----------------------------------------------------------------
