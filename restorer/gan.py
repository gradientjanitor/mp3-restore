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
        self.twodim_layers = nn.ModuleList()

        # 2 dim layer -- looks at spectrogram as an image.
        dummy_2d = dummy.reshape(dummy.shape[0], 1, dummy.shape[1], dummy.shape[2])
        prev_chan = 1
        for l in [64,128,256]:
            self.twodim_layers.append(nn.Conv2d(prev_chan, l, 5))
            dummy_2d = self.twodim_layers[-1](dummy_2d)
            self.twodim_layers.append(nn.InstanceNorm2d(l, affine=True))
            self.twodim_layers.append(nn.MaxPool2d(2))
            dummy_2d = self.twodim_layers[-1](dummy_2d)


            self.twodim_layers.append(self.act)
            prev_chan = l

        print("two dim:", dummy_2d.shape)

        # one dim layer -- looks at full spectral content for each timestep.
        prev_chan = fft_dim
        for l in range(len(num_chan)):
            curr_chan = num_chan[l]
            self.layers.append(nn.Conv1d(prev_chan, curr_chan, kernel_sz, padding=kernel_sz//2, stride=1, padding_mode='reflect'))
            dummy = self.layers[-1](dummy)
            self.layers.append(nn.InstanceNorm1d(curr_chan, affine=True))
            dummy = self.layers[-1](dummy)

            #self.layers.append(nn.MaxPool1d(2))
            #dummy = self.layers[-1](dummy)
            
            self.layers.append(self.act)
            dummy = self.layers[-1](dummy)
            prev_chan = curr_chan

        print(dummy.shape)
        self.out_layer = nn.Linear(dummy.shape[1] * dummy.shape[2] + dummy_2d.shape[1], 1)
        self.apply(weight_init)

        print(self)

    def forward(self, recon, compr):
        # x = torch.cat([recon, compr], dim=1)
        x = recon
        x_2d = x.reshape(x.shape[0], 1, x.shape[1], x.shape[2])

        for l in self.twodim_layers:
            x_2d = l(x_2d)
        x_2d = x_2d.view(x_2d.shape[0], x_2d.shape[1], -1)
        x_2d = torch.max(x_2d, dim=2)[0]

        for l in self.layers:
            x = l(x)

        x = x.view(x.shape[0], -1)
        x = self.out_layer(torch.cat([x, x_2d], dim=1))

        return x

class ReconNet(nn.Module):
    def __init__(self, fft_dim, kernel_sz=25, num_chan=[128,256,512,256,128], add_noise=False, residual=False):
        super(ReconNet, self).__init__()

        self.act = nn.ELU()

        self.layers = nn.ModuleList()
        self.add_noise = add_noise
        self.residual = residual

        if add_noise:
            prev_chan = fft_dim * 2
            num_chan *= 2
        else:
            prev_chan = fft_dim

        for l in range(len(num_chan)):
            curr_chan = num_chan[l]
            self.layers.append(nn.Conv1d(prev_chan, curr_chan, kernel_sz, stride=1, padding=kernel_sz//2, padding_mode='reflect'))
            self.layers.append(nn.BatchNorm1d(curr_chan))
            self.layers.append(self.act)
            prev_chan = curr_chan

        self.mag_out = nn.Conv1d(prev_chan, fft_dim // 2, kernel_sz, stride=1, padding=kernel_sz//2, padding_mode='reflect')
        self.phase_out = nn.Conv1d(prev_chan, fft_dim // 2, kernel_sz, stride=1, padding=kernel_sz//2, padding_mode='reflect')

        self.apply(weight_init)

        print(self)

    def forward(self, x):
        xi = x
        if self.add_noise:
            xn = torch.randn_like(x)
            x = torch.cat([xi, xn], dim=1)

        for l in self.layers:
            x = l(x)

        xm = self.mag_out(x)
        xp = self.phase_out(x)

        xp = 2 * torch.sigmoid(xp) - 1

        x = torch.cat([xm, xp], dim=1)

        if self.residual:
            return x + xi
        else:
            return x
