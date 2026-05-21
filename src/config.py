# ============================================================
# Central configuration (important for experiments)
# ============================================================
from datetime import datetime
from pathlib import Path

import torch

DATA_DIR_PATH = "./data/"
EXP_NAME = "amp_tcn"
timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
RUN_DIR = f"runs/{timestamp}_{EXP_NAME}"
Path(RUN_DIR).mkdir(parents=True, exist_ok=True)
DI_PATH = DATA_DIR_PATH + "input.wav"
AMP_PATH = DATA_DIR_PATH + "output.wav"

BATCH_SIZE = 16
EPOCHS = 20
LR = 1e-3

SEGMENT_LENGTH = 16384

CHANNELS = 16
DILATIONS = [1, 2, 4, 8, 16, 32, 64]
STACKS = 2

# Device selection (MPS/CUDA safe)

if torch.cuda.is_available():
    DEVICE = "cuda"
elif torch.backends.mps.is_available():
    DEVICE = "mps"
else:
    DEVICE = "cpu"
