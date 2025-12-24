import os
from pathlib import Path
import numpy as np
import pytest
from .data_module import *

def generate_dummy_token_file(
    file_path: Union[str, Path],
    num_tokens: int,
    vocab_size: int = 1000,
    seed: Optional[int] = 42,
) -> Path:
    p = Path(file_path)
    if seed is not None:
        rng = np.random.default_rng(seed)
    else:
        rng = np.random.default_rng()
    p.parent.mkdir(parents=True, exist_ok=True)
    tokens = rng.integers(low=0, high=vocab_size, size=num_tokens, dtype=np.int32)
    tokens.tofile(p)
    return p


@pytest.mark.parametrize("max_length,num_sequences", [(8, 3), (16, 2)])
def test_tokenized_dataset_len_and_item(tmp_path: Path, max_length, num_sequences):
    torch = pytest.importorskip("torch")

    total_tokens = max_length * num_sequences + 1
    bin_path = tmp_path / "data.bin"
    
    tokens = np.arange(total_tokens, dtype=np.int32)
    tokens.tofile(bin_path)

    ds = TokenizedDataset(bin_path, max_length=max_length)
    assert len(ds) == num_sequences

    item0 = ds[0]
    input_ids = item0["input_ids"]
    labels = item0["labels"]
    assert input_ids.shape[0] == max_length
    assert labels.shape[0] == max_length
    assert torch.equal(labels, input_ids + 1)

    with pytest.raises(IndexError):
        _ = ds[len(ds)]



@pytest.mark.parametrize(
    "max_length,num_sequences,batch_size,drop_last",
    [
        (8, 5, 2, False),
        (8, 5, 2, True),
        (8, 4, 4, False),
    ],
)
def test_dataloader_batching(tmp_path: Path, max_length, num_sequences, batch_size, drop_last):
    torch = pytest.importorskip("torch")

    total_tokens = max_length * num_sequences + 1
    bin_path = tmp_path / "data.bin"

    generate_dummy_token_file(bin_path, num_tokens=total_tokens, vocab_size=100, seed=123)

    ds = TokenizedDataset(bin_path, max_length=max_length)
    loader = DataLoader(ds, batch_size=batch_size, shuffle=False, drop_last=drop_last)

    batches = loader

    if drop_last:
        expected_batches = (len(ds) // batch_size)
    else:
        expected_batches = (len(ds) + batch_size - 1) // batch_size

    assert len(batches) == expected_batches

    for i, batch in enumerate(batches):
        assert isinstance(batch, dict)
        assert "input_ids" in batch and "labels" in batch

        x, y = batch["input_ids"], batch["labels"]
        assert torch.is_tensor(x) and torch.is_tensor(y)
        assert x.shape[1] == max_length
        assert y.shape[1] == max_length

        expected_b = batch_size if (i < len(batches) - 1 or not drop_last) else x.shape[0]
        if i < len(batches) - 1:
            assert x.shape[0] == batch_size
        else:
            if drop_last:
                assert x.shape[0] == batch_size
            else:
                assert 1 <= x.shape[0] <= batch_size


def test_dataloader_shuffle_seed_repro(tmp_path: Path):
    torch = pytest.importorskip("torch")

    max_length = 6
    num_sequences = 7
    total_tokens = max_length * num_sequences + 1
    bin_path = tmp_path / "data.bin"
    
    tokens = np.arange(total_tokens, dtype=np.int32)
    tokens.tofile(bin_path)

    ds = TokenizedDataset(bin_path, max_length=max_length)

    ldr1 = DataLoader(ds, batch_size=2, shuffle=True, seed=777)
    ldr2 = DataLoader(ds, batch_size=2, shuffle=True, seed=777)

    def flatten_batches(loader):
        order = []
        for batch in loader:
            order.extend(batch["input_ids"][:, 0].tolist())
        return order

    order1 = flatten_batches(ldr1)
    order2 = flatten_batches(ldr2)
    assert order1 == order2, "the same seed should produce the same order"

    ldr3 = DataLoader(ds, batch_size=2, shuffle=True, seed=778)
    order3 = flatten_batches(ldr3)
    ldr4 = DataLoader(ds, batch_size=2, shuffle=True, seed=779)
    order4 = flatten_batches(ldr4)
    assert not (order1 == order3 == order4), "diffrent seed should produce different order"
    