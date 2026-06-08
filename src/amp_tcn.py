import torch.nn as nn
import torch.nn.functional as F


class CausalConv1d(nn.Module):
    def __init__(self, channels, kernel_size, dilation):
        super().__init__()
        self.pad = (kernel_size - 1) * dilation
        self.conv = nn.Conv1d(channels, channels, kernel_size, dilation=dilation)

    def forward(self, x):
        return self.conv(F.pad(x, (self.pad, 0)))


class AmpTCN(nn.Module):
    def __init__(self, channels, dilations, stacks):
        super().__init__()

        layers = [nn.Conv1d(1, channels, 1)]
        for _ in range(stacks):
            for d in dilations:
                layers += [CausalConv1d(channels, 3, d), nn.PReLU()]
        layers.append(nn.Conv1d(channels, 1, 1))

        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)
