# ============================================================
# Central configuration (important for experiments)
# ============================================================
from datetime import datetime
from pathlib import Path

import torch


# DEFINE experiment name
EXP_NAME = "amp_tcn_100_epochs_128_batch_size_exponentialLR_80_20_esr_mrstft_ReduceLROnPlateau_4stacks_acutally"


# HYPERPARAMETERS HERE
BATCH_SIZE = 128
EPOCHS = 100
LR = 1e-3

SEGMENT_LENGTH = 16384

CHANNELS = 16
DILATIONS = [1, 2, 4, 8, 16, 32, 64, 128]
STACKS = 2
#####################

DATA_DIR_PATH = "./data/"

timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

RUN_DIR = f"runs/{timestamp}_{EXP_NAME}"

Path(RUN_DIR).mkdir(parents=True, exist_ok=True)

DI_PATH = DATA_DIR_PATH + "input.wav"

AMP_PATH = DATA_DIR_PATH + "output.wav"


# Device selection (MPS/CUDA safe)
if torch.cuda.is_available():
    DEVICE = "cuda"
elif torch.backends.mps.is_available():
    DEVICE = "mps"
else:
    DEVICE = "cpu"
