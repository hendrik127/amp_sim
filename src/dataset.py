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
    """

    def __init__(self, di_path, amp_path, segment_length=16384, dataset_size=20000):
        self.segment_length = segment_length
        self.dataset_size = dataset_size

        self.di, sr1 = load_mono_audio(di_path)
        self.amp, sr2 = load_mono_audio(amp_path)

        assert sr1 == sr2, "Sample rate mismatch!"

        min_len = min(len(self.di), len(self.amp))
        self.di = self.di[:min_len]
        self.amp = self.amp[:min_len]

    def __len__(self):
        return self.dataset_size

    def __getitem__(self, idx):
        max_start = len(self.di) - self.segment_length
        start = random.randint(0, max_start)

        end = start + self.segment_length

        x = self.di[start:end]
        y = self.amp[start:end]

        x = torch.from_numpy(x).unsqueeze(0)
        y = torch.from_numpy(y).unsqueeze(0)

        return x, y
