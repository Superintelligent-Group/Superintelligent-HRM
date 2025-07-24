import torch
from torch.utils.data import Dataset
import numpy as np
import os
import json

class LanguageDataset(Dataset):
    """
    A highly efficient Dataset for reading pre-tokenized binary files.
    It uses memory-mapping for near-instantaneous loading of large datasets.
    """
    def __init__(self, data_dir: str, split: str, block_size: int):
        super().__init__()
        self.block_size = block_size
        
        # Load the tokenized data using memory-mapping for efficiency
        bin_path = os.path.join(data_dir, f"{split}.bin")
        self.data = np.memmap(bin_path, dtype=np.uint16, mode='r')
        
        print(f"Loaded {split} data with {len(self.data)} tokens.")

    def __len__(self):
        # The number of possible starting points for a block
        return len(self.data) - self.block_size

    def __getitem__(self, idx):
        # Grab a chunk of tokens of size block_size
        chunk = self.data[idx : idx + self.block_size]
        
        # The input to the model is the sequence, the target is the sequence shifted by one
        x = torch.from_numpy(chunk.astype(np.int64))
        y = torch.from_numpy(self.data[idx + 1 : idx + self.block_size + 1].astype(np.int64))
        
        return x, y

def load_metadata(data_dir: str):
    """Loads the metadata file."""
    meta_path = os.path.join(data_dir, 'meta.json')
    if not os.path.exists(meta_path):
        return None
    with open(meta_path, 'r') as f:
        return json.load(f) 