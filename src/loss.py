import auraloss
import torch


class CombinedLoss(torch.nn.Module):
    def __init__(self):
        super().__init__()

        self.mrstft = auraloss.freq.MultiResolutionSTFTLoss(
            fft_sizes=[64, 128, 256, 512, 1024],
            hop_sizes=[16, 32, 64, 128, 256],
            win_lengths=[64, 128, 256, 512, 1024],
            w_sc=0.0,
            w_log_mag=1.0,
        )

    def forward(self, pred, target):
        eps = 1e-8

        # -------------------------
        # ESR (time domain)
        # -------------------------
        esr = torch.sum((pred - target) ** 2) / (torch.sum(target**2) + eps)

        # -------------------------
        # Spectral loss
        # -------------------------
        spec = self.mrstft(pred, target)

        # -------------------------
        # Combined
        # -------------------------
        total = 0.8 * esr + 0.2 * spec

        return total, esr, spec
