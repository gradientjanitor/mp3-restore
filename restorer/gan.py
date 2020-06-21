import torch

from torch import nn
from torch import optim

def weight_init(m):
    if isinstance(m, torch.nn.Conv1d) or isinstance(m, torch.nn.Linear) or isinstance(m, torch.nn.Conv2d):
        print(m)
        torch.nn.init.orthogonal_(m.weight)
        torch.nn.init.zeros_(m.bias)
    elif isinstance(m, torch.nn.InstanceNorm1d) or isinstance(m, torch.nn.BatchNorm1d):
        print(m)
        torch.nn.init.ones_(m.weight)
        torch.nn.init.zeros_(m.bias)


class DiscrNet(nn.Module):
    def __init__(self, dummy, fft_dim, kernel_sz=25, num_chan=[512,256,128]):
        super(DiscrNet, self).__init__()

        self.act = nn.ELU()

        self.layers = nn.ModuleList()

        dummy = dummy.view(dummy.shape[0], 1, dummy.shape[1], dummy.shape[2])
        dummy = torch.cat([dummy, dummy, dummy], dim=1)
        # one dim layer -- looks at full spectral content for each timestep.
        prev_chan = 3 #fft_dim
        for l in range(len(num_chan)):
            curr_chan = num_chan[l]
            self.layers.append(nn.Conv2d(prev_chan, curr_chan, kernel_sz, padding=kernel_sz//2, stride=1, padding_mode='reflect'))
            dummy = self.layers[-1](dummy)

            self.layers.append(nn.BatchNorm2d(curr_chan, affine=True))
            self.layers.append(self.act)
            self.layers.append(nn.AvgPool2d(2))
            dummy = self.layers[-1](dummy)

            prev_chan = curr_chan

        print(dummy.shape)
        self.out_layers = nn.ModuleList()
        self.out_layers.append(nn.Linear(dummy.shape[1] * dummy.shape[2] * dummy.shape[3], 1))

        self.apply(weight_init)

        print(self)

    def forward(self, x1, x2, xc):
        # x: the reconstructed or high-bitrate spectrogram
        # xc: the compressed spectrogram
        x1 = x1.view(x1.shape[0], 1, x1.shape[1], x1.shape[2])
        x2 = x2.view(x2.shape[0], 1, x2.shape[1], x2.shape[2])
        xc = xc.view(xc.shape[0], 1, xc.shape[1], xc.shape[2])

        x = torch.cat([x1, x2, xc], dim=1)

        for l in self.layers:
            x = l(x)

        x = x.view(x.shape[0], -1)
        for l in self.out_layers:
            x = l(x)

        return x

class Residual(nn.Module):
    def __init__(self, dim, kernel_sz):
        super(Residual, self).__init__()
        self.conv = nn.Conv1d(dim, dim, kernel_sz, padding=kernel_sz//2, padding_mode='reflect')
        self.bn = nn.BatchNorm1d(dim)
        self.act = nn.ELU()

    def forward(self, x):
        return self.act(self.bn(self.conv(x)))

class ReconNet(nn.Module):
    def __init__(self, fft_dim, kernel_sz=25, num_layers=10, residual=False):
        super(ReconNet, self).__init__()

        self.act = nn.ELU()

        self.mag_layers = nn.ModuleList()
        self.residual = residual

        prev_chan = fft_dim

        for l in range(num_layers):
            self.mag_layers.append(Residual(fft_dim, kernel_sz))

        self.out_layer = nn.Conv1d(fft_dim, fft_dim, kernel_sz, padding=kernel_sz//2, padding_mode='reflect')
        self.apply(weight_init)

        print(self)

    def forward(self, x):
        xi = x

        for l in self.mag_layers:
            x = l(x) + x

        x = self.out_layer(x)

        return torch.tanh(x)
