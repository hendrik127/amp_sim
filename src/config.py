from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from abc import ABC, abstractmethod

import torch
'''
DI_PATH = Path("../data/input.wav")
AMP_PATH = Path("../data/output.wav")
DI_VAL_PATH = Path("../data/input_val.wav")
AMP_VAL_PATH = Path("../data/output_val.wav")
'''
PROJECT_ROOT = Path(__file__).parent.parent

DI_PATH = PROJECT_ROOT / "data" / "input.wav"
AMP_PATH = PROJECT_ROOT / "data" / "output.wav"
DI_VAL_PATH = PROJECT_ROOT / "data" / "input_val.wav"  
AMP_VAL_PATH = PROJECT_ROOT / "data" / "output_val.wav"

if torch.cuda.is_available():
    DEVICE = "cuda"
elif torch.backends.mps.is_available():
    DEVICE = "mps"
else:
    DEVICE = "cpu"

@dataclass
class BaseConfig(ABC):
    # Training
    batch_size: int = 128
    epochs: int = 100
    lr: float = 1e-3
    segment_length: int = 16384
    train_dataset_size: int = 20000 #samples per epoch
    # Scheduler: "ReduceLROnPlateau" | "ExponentialLR" | "CosineAnnealingLR"
    scheduler: str = "ReduceLROnPlateau"
    scheduler_kwargs: dict = field(default_factory=dict)

    @abstractmethod
    def slug(self) -> str:
        #Unique identifier for this experiment
        pass
    
    @abstractmethod
    def create_model(self):
        #Make the model
        pass

    def make_run_dir(self) -> Path:
        ts = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        p = Path(f"runs/{ts}_{self.slug()}")
        p.mkdir(parents=True, exist_ok=True)
        return p
    
    def to_dict(self) -> dict:
        return asdict(self)
    

@dataclass
class TCNConfig(BaseConfig):
    channels: int = 16
    dilations: list = field(default_factory=lambda: [1, 2, 4, 8, 16, 32, 64, 128])
    stacks: int = 2

    def slug(self) -> str:
        sched = self.scheduler.replace("LR", "").replace("OnPlateau", "")
        return f"ch{self.channels}_s{self.stacks}_dil{max(self.dilations)}_lr{self.lr:.0e}_{sched}"
    
    def create_model(self):
        from models.AmpTCN import AmpTCN  
        return AmpTCN(self.channels, self.dilations, self.stacks)

@dataclass
class GRUConfig(BaseConfig):
    hidden_size: int = 32
    num_layers: int = 2
    dropout: float = 0.1
    
    def slug(self) -> str:
        sched = self.scheduler.replace("LR", "").replace("OnPlateau", "")
        return f"gru_h{self.hidden_size}_l{self.num_layers}_d{self.dropout}_lr{self.lr:.0e}_{sched}"

    def create_model(self):
        from models.AmpGRU import AmpGRU
        return AmpGRU(1, self.hidden_size, self.num_layers, self.dropout)

# ── Define experiments here ──────────────────────────────────────────────────

EXPERIMENTS = [
    # --- baseline ---
    TCNConfig(),  # ch16 s2 ~12.6K params
    # --- smaller ---
    TCNConfig(channels=8, stacks=2),  # ch8  s2  ~3.2K params
    TCNConfig(channels=8, stacks=1),  # ch8  s1  ~1.6K params
    TCNConfig(channels=16, stacks=1),  # ch16 s1  ~6.3K params
    # --- wider single stack ---
    TCNConfig(channels=32, stacks=1),  # ch32 s1  ~24.9K params
    # --- scheduler variants (baseline arch) ---
    TCNConfig(
        scheduler="ExponentialLR",
        scheduler_kwargs={"gamma": 0.995},
    ),
    TCNConfig(
        scheduler="CosineAnnealingLR",
    ),
]
