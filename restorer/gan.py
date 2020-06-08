import torch

from torch import nn
from torch import optim

def weight_init(m):
    if isinstance(m, torch.nn.Conv1d) or isinstance(m, torch.nn.Linear):
        print(m)
        torch.nn.init.xavier_uniform_(m.weight)
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
        # one dim layer -- looks at full spectral content for each timestep.
        prev_chan = 1 #fft_dim
        for l in range(len(num_chan)):
            curr_chan = num_chan[l]
            self.layers.append(nn.Conv2d(prev_chan, curr_chan, kernel_sz, padding=kernel_sz//2, stride=1, padding_mode='reflect'))
            dummy = self.layers[-1](dummy)

            self.layers.append(nn.InstanceNorm2d(curr_chan, affine=True))
            self.layers.append(self.act)
            self.layers.append(nn.MaxPool2d(2))
            dummy = self.layers[-1](dummy)

            prev_chan = curr_chan

        print(dummy.shape)
        self.out_layers = nn.ModuleList()
        self.out_layers.append(nn.Linear(dummy.shape[1] * dummy.shape[2] * dummy.shape[3], 1))

        self.apply(weight_init)

        print(self)

    def forward(self, x):
        x = x.view(x.shape[0], 1, x.shape[1], x.shape[2])

        for l in self.layers:
            x = l(x)

        x = x.view(x.shape[0], -1)
        for l in self.out_layers:
            x = l(x)

        return x

class ReconNet(nn.Module):
    def __init__(self, fft_dim, kernel_sz=25, num_chan=[128,256,512,256,128], residual=False):
        super(ReconNet, self).__init__()

        self.act = nn.ELU()

        self.mag_layers = nn.ModuleList()
        self.residual = residual

        prev_chan = 0

        for l in range(len(num_chan)):
            curr_chan = num_chan[l]
            self.mag_layers.append(nn.Conv1d(prev_chan + fft_dim, curr_chan, kernel_sz, stride=1, padding=kernel_sz//2, padding_mode='reflect'))
            self.mag_layers.append(nn.BatchNorm1d(curr_chan))
            self.mag_layers.append(self.act)
            self.mag_layers.append(nn.Dropout())

            prev_chan = curr_chan
            
        self.mag_layers.append(nn.Conv1d(prev_chan + fft_dim, fft_dim, kernel_sz, stride=1, padding=kernel_sz//2, padding_mode='reflect'))

        self.apply(weight_init)

        print(self)

    def forward(self, x):
        xi = x

        first = True
        for l in self.mag_layers:
            if self.residual and not first and isinstance(l, torch.nn.Conv1d):
                x = torch.cat([x, xi], dim=1)

            x = l(x)
            first = False

        return x
