import random

import torch
from torch.utils.data import Dataset

from utils import load_mono_audio


class LongAudioDataset(Dataset):
    """
    Single-file DI + amp training dataset.

    Key idea:
    - load full recording once
    - randomly sample chunks forever

    Sample randomly during training, deterministically during validation
    """

    def __init__(self, di_path, amp_path, segment_length=16384, dataset_size=20000, is_val=False):
        self.segment_length = segment_length
        self.dataset_size = dataset_size
        self.is_val = is_val

        self.di, sr1 = load_mono_audio(di_path)
        self.amp, sr2 = load_mono_audio(amp_path)

        assert sr1 == sr2, "Sample rate mismatch!"

        min_len = min(len(self.di), len(self.amp))
        self.di = self.di[:min_len]
        self.amp = self.amp[:min_len]

        if is_val:
            self.chunks = []
            start = 0
            while start + segment_length <= len(self.di):
                self.chunks.append((start, start + segment_length))
                start += segment_length  
        else:
            self.dataset_size = dataset_size

        

    def __len__(self):
        return len(self.chunks) if self.is_val else self.dataset_size

    def __getitem__(self, idx):
        if self.is_val:
            start, end = self.chunks[idx]
        else:
            max_start = len(self.di) - self.segment_length
            start = random.randint(0, max_start)
            end = start + self.segment_length
        
        x = torch.from_numpy(self.di[start:end]).unsqueeze(0)
        y = torch.from_numpy(self.amp[start:end]).unsqueeze(0)
        return x, y
