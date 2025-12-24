import os
import glob
import numpy as np
import pyarrow.parquet as pq
from transformers import AutoTokenizer
from tqdm import tqdm
from multiprocessing import Pool
import time

MODEL_PATH = "/thullms/public/Qwen/Qwen3-0.6B-Base"
DATA_PATH = "/thullms/public/HuggingFaceFW/fineweb-edu-10BT/sample/10BT"
OUTPUT_DIR = "/thullms/3022377347/tokenized_fineweb_edu_10b"

os.makedirs(OUTPUT_DIR, exist_ok=True)

_tokenizer = None
_eot_id = None

def init_worker(model_path):
    global _tokenizer, _eot_id
    _tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=True)
    _eot_id = _tokenizer.eos_token_id
    if _eot_id is None:
        _eot_id = _tokenizer.convert_tokens_to_ids("<|endoftext|>")
    print(f"Worker {os.getpid()} initialized")

def tokenize_batch(texts):
    global _tokenizer, _eot_id
    out = []
    for text in texts:
        ids = _tokenizer.encode(text, add_special_tokens=False)
        ids.append(_eot_id)
        out.extend(ids)
    return out

def main():
    print("=" * 60)
    print("FineWeb-Edu 10BT Tokenization")
    print("=" * 60)

    print("\nLoading Qwen3 tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, use_fast=True)
    eot_id = tokenizer.eos_token_id or tokenizer.convert_tokens_to_ids("<|endoftext|>")
    print(f"EOT token ID: {eot_id}")

    parquet_files = sorted(glob.glob(os.path.join(DATA_PATH, "*.parquet")))
    print(f"\nFound {len(parquet_files)} parquet files")

    VAL_TOKENS = 20_000_000
    NUM_PROC = 14
    BATCH_SIZE = 1000

    train_path = os.path.join(OUTPUT_DIR, "train.bin")
    val_path = os.path.join(OUTPUT_DIR, "val.bin")

    train_f = open(train_path, "wb")
    val_f = open(val_path, "wb")

    total_tokens_seen = 0
    total_docs = 0
    start_time = time.time()

    print(f"\nStarting tokenization with {NUM_PROC} workers...")
    print(f"Batch size: {BATCH_SIZE}")
    print(f"Validation tokens: {VAL_TOKENS:,}")

    for file_idx, file_path in enumerate(parquet_files, 1):
        file_start_time = time.time()

        table = pq.read_table(file_path)
        texts = table['text'].to_pylist()
        total_docs += len(texts)

        batches = [texts[i:i + BATCH_SIZE] for i in range(0, len(texts), BATCH_SIZE)]
        rng = np.random.default_rng()
        rng.shuffle(batches)

        print(f"\n[{file_idx}/{len(parquet_files)}] Processing {os.path.basename(file_path)}")
        print(f"  Documents: {len(texts):,} | Batches: {len(batches)}")

        with Pool(NUM_PROC, initializer=init_worker, initargs=(MODEL_PATH,)) as pool:
            for out in tqdm(
                pool.imap_unordered(tokenize_batch, batches),
                total=len(batches),
                desc=f"  Tokenizing",
                ncols=80
            ):
                if total_tokens_seen < VAL_TOKENS:
                    need = VAL_TOKENS - total_tokens_seen
                    if len(out) <= need:
                        val_f.write(np.array(out, dtype=np.uint32).tobytes())
                        total_tokens_seen += len(out)
                        continue
                    val_f.write(np.array(out[:need], dtype=np.uint32).tobytes())
                    rest = out[need:]
                    total_tokens_seen += need
                else:
                    rest = out

                if rest:
                    train_f.write(np.array(rest, dtype=np.uint32).tobytes())
                    total_tokens_seen += len(rest)

        file_elapsed = time.time() - file_start_time
        print(f"  Completed in {file_elapsed:.1f}s")
        print(f"  Total tokens so far: {total_tokens_seen:,}")

        elapsed = time.time() - start_time
        tps = total_tokens_seen / elapsed if elapsed > 0 else 0
        est_total = elapsed / file_idx * len(parquet_files)
        eta = est_total - elapsed

        print(f"  Speed: {tps:,.0f} tokens/sec")
        print(f"  Elapsed: {elapsed/3600:.2f}h | ETA: {eta/3600:.2f}h")

    train_f.close()
    val_f.close()

    total_time = time.time() - start_time
    val_tokens = min(total_tokens_seen, VAL_TOKENS)
    train_tokens = total_tokens_seen - val_tokens

    print("\n" + "=" * 60)
    print("✅ Tokenization Complete!")
    print("=" * 60)
    print(f"Total time: {total_time/3600:.2f} hours")
    print(f"Total documents: {total_docs:,}")
    print(f"Total tokens: {total_tokens_seen:,}")
    print(f"  - Validation: {val_tokens:,} tokens ({val_tokens/1e6:.1f}M)")
    print(f"  - Training: {train_tokens:,} tokens ({train_tokens/1e9:.2f}B)")
    print(f"Average speed: {total_tokens_seen/total_time:,.0f} tokens/sec")
    print(f"\nOutput files:")
    print(f"  {val_path}")
    print(f"  {train_path}")

    val_size = os.path.getsize(val_path) / (1024**3)
    train_size = os.path.getsize(train_path) / (1024**3)
    print(f"\nFile sizes:")
    print(f"  val.bin: {val_size:.2f} GB")
    print(f"  train.bin: {train_size:.2f} GB")

if __name__ == "__main__":
    main()
