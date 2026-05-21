import torch.nn as nn


class CausalConv1d(nn.Module):
    def __init__(self, in_ch, out_ch, k, dilation=1):
        super().__init__()

        self.pad = (k - 1) * dilation

        self.conv = nn.Conv1d(
            in_ch,
            out_ch,
            kernel_size=k,
            dilation=dilation,
            padding=self.pad,
        )

    def forward(self, x):
        y = self.conv(x)
        return y[:, :, : -self.pad] if self.pad > 0 else y


class ResidualBlock(nn.Module):
    def __init__(self, channels, dilation):
        super().__init__()

        self.conv = CausalConv1d(channels, channels, 3, dilation)
        self.act = nn.PReLU()
        self.mix = nn.Conv1d(channels, channels, 1)

    def forward(self, x):
        r = x
        x = self.conv(x)
        x = self.act(x)
        x = self.mix(x)
        return x + r


class AmpTCN(nn.Module):
    def __init__(self, channels, dilations, stacks):
        super().__init__()

        self.in_conv = nn.Conv1d(1, channels, 1)

        blocks = []
        for _ in range(stacks):
            for d in dilations:
                blocks.append(ResidualBlock(channels, d))

        self.blocks = nn.ModuleList(blocks)

        self.out_conv = nn.Conv1d(channels, 1, 1)

    def forward(self, x):
        x = self.in_conv(x)
        for b in self.blocks:
            x = b(x)
        return self.out_conv(x)
