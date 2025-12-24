import os
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from typing import List, Union, Optional
from pathlib import Path
from collections.abc import Mapping
import random

class TokenizedDataset(Dataset):
    def __init__(self, file_path: Union[str, Path], max_length: int):
        self.file_path = Path(file_path)
        self.max_length = max_length

        self.data = np.fromfile(self.file_path, dtype=np.int32)
        self.num_samples = len(self.data) // self.max_length
    def __len__(self):
        return self.num_samples
    
    def __getitem__(self, idx):
        if idx < 0 or idx >= len(self):
            raise IndexError("Index out of range")
        start_idx = idx * self.max_length
        end_idx = start_idx + self.max_length
        sample = self.data[start_idx:end_idx + 1]
        return {
            "input_ids": torch.from_numpy(sample[:-1]),
            "labels": torch.from_numpy(sample[1:])
        }


class DataLoader:
    def __init__(self,
        dataset: TokenizedDataset,
        batch_size: int,
        shuffle: bool,
        drop_last: bool = False,
        collate_fn = None,
        seed: Optional[int] = None):
        self.dataset = dataset
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.drop_last = drop_last
        self.collate_fn = collate_fn if collate_fn is not None else self.default_collate
        self.seed = seed
        self.indices = list(range(len(self.dataset)))
        if self.shuffle:
            if self.seed is not None:
                random.seed(self.seed)
            random.shuffle(self.indices)
    
    def __len__(self) -> int:
        if self.drop_last:
            return len(self.indices) // self.batch_size
        else:
            return (len(self.indices) + self.batch_size - 1) // self.batch_size
    
    def default_collate(self, batch: List[Mapping]) -> Mapping:
        collated = {}
        for key in batch[0]:
            collated[key] = torch.stack([item[key] for item in batch], dim=0)
        return collated

    def __iter__(self):
        for start_idx in range(0, len(self.indices), self.batch_size):
            end_idx = start_idx + self.batch_size
            if end_idx > len(self.indices) and self.drop_last:
                break
            batch_indices = self.indices[start_idx:end_idx]
            batch = [self.dataset[i] for i in batch_indices]
            yield self.collate_fn(batch)

        
    