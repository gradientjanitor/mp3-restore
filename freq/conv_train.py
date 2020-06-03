import glob
import time

from scipy.io import wavfile
from scipy.signal import stft, istft
import torch
import random

import numpy as np

from torch import nn
from torch import optim

from matplotlib import pyplot as plt

def weight_init(m):
    if isinstance(m, torch.nn.Conv1d) or isinstance(m, torch.nn.Linear):
        print(m)
        torch.nn.init.xavier_uniform_(m.weight)
        torch.nn.init.zeros_(m.bias)
    elif isinstance(m, torch.nn.InstanceNorm1d) or isinstance(m, torch.nn.BatchNorm1d):
        print(m)
        torch.nn.init.ones_(m.weight)
        torch.nn.init.zeros_(m.bias)

def detach_numpy(x):
    return x.detach().cpu().numpy()


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


def get_stft(x, fft_len):
    xf = stft(x.flatten(), nperseg=fft_len, return_onesided=True)[2]

    xm = np.log(1e-10+np.abs(xf)) + 8
    xp = np.angle(xf)
    xp /= np.pi

    xmp = np.concatenate([xm, xp], axis=0)

    return xmp


def get_istft(x):
    fft_dim = x.shape[0] // 2
    xm = np.exp(x[:fft_dim,:] - 8)
    xp = x[fft_dim:,:] * np.pi
    xf = xm * np.exp(1j * xp)

    xr = istft(xf, input_onesided=True)[1]

    return xr.flatten()


def load_dataset(data_dir, fft_len=512, nsamp=None):
    sample_pairs = []

    files = glob.glob("%s/*_high.wav" % data_dir)
    if nsamp is not None:
        files = files[:1]

    for f in files:
        f_compr = f.replace("_high.wav", "_low.wav")
        orig_sample = (wavfile.read(f)[1].astype(np.float32) / 32768)
        compr_sample = (wavfile.read(f_compr)[1].astype(np.float32) / 32768)

        compr_stft_l = get_stft(compr_sample[:,0:1], fft_len)
        orig_stft_l = get_stft(orig_sample[:,0:1], fft_len)
        compr_stft_r = get_stft(compr_sample[:,1:2], fft_len)
        orig_stft_r = get_stft(orig_sample[:,1:2], fft_len)

        sample_pairs.append([compr_stft_l, orig_stft_l, compr_stft_r, orig_stft_r, f])

    return sample_pairs


def make_batch(sample_pairs, datadim, nsteps=64, cuda=False):
    x = np.zeros((len(sample_pairs), datadim, nsteps), dtype=np.float32)
    y = np.zeros((len(sample_pairs), datadim, nsteps), dtype=np.float32)

    for i, sample in enumerate(sample_pairs):
        compr_sample_l, orig_sample_l, compr_sample_r, orig_sample_r, filename = sample


        # choose random channel
        if np.random.rand() < 0.5:
            compr_sample, orig_sample = compr_sample_l, orig_sample_l
        else:
            compr_sample, orig_sample = compr_sample_r, orig_sample_r

        sstart = np.random.randint(0, compr_sample.shape[1] - nsteps - 1)
        compr_sample = compr_sample[:,sstart:sstart+nsteps]
        orig_sample = orig_sample[:,sstart:sstart+nsteps]

        x[i,:,:] = compr_sample
        y[i,:,:] = orig_sample

    x, y = torch.from_numpy(x), torch.from_numpy(y)
    if cuda:
        x, y = x.cuda(), y.cuda()

    return x, y


data_dir = "../data/raw"

best_model = None
best_loss = 100

for midx in range(15):
    # fft_len = random.choice([256,512,784,1024])
    fft_len = random.choice([256,512])
    kernel_sz = random.choice([3,5,7,9,11])
    lr = random.choice([0.0001])
    nsteps = random.choice([32,48,64])
    datadim = fft_len + 2

    add_noise = False #random.choice([True,False])
    residual = True #random.choice([True,False])
    num_chan_proto = random.choice([[1,2,3],[1,2,3,2,1],[1,2,3,4,5]])
    num_chan = np.asarray(num_chan_proto) * datadim #random.choice([128,256,512])

    model_str = " ".join([str(s) for s in [fft_len, kernel_sz, lr, nsteps,  num_chan, add_noise, residual]])
    
    print("generating stfts...")
    sample_pairs = load_dataset(data_dir, fft_len)
    x, y = make_batch(sample_pairs, datadim, nsteps=nsteps, cuda=True)
    print("done.")

    gen = ReconNet(datadim, kernel_sz=kernel_sz, num_chan=num_chan, add_noise=add_noise, residual=residual).cuda()
    discr = DiscrNet(x.cpu(), datadim, kernel_sz=9, num_chan=[512,256,128,64]).cuda()
    gen_opt = optim.Adam(gen.parameters(), lr=lr)
    discr_opt = optim.Adam(discr.parameters(), lr=lr)
    
    t = time.time()

    loss_avg = []


    discr_losses = []
    gen_gan_losses= []
    gen_mse_losses= []

    i = 0
    while True:
        i += 1
        # timeout after 15min of training.
        #if time.time() - t > 60 * 60:
        #    break

        x, y = make_batch(sample_pairs, datadim, nsteps=nsteps, cuda=True)
    
        gen.train()
        discr.train()
    
        try:
            gen_opt.zero_grad()
            discr_opt.zero_grad()

            gen_out = gen.forward(x)

            # train generator
            discr_gen_out = discr.forward(gen_out, x)

            # cross entropy
            gen_gan_loss = nn.BCEWithLogitsLoss()(discr_gen_out, torch.ones(gen_out.shape[0], 1).cuda())
            gen_mse_loss = torch.mean(torch.abs(gen_out - y))

            gen_loss = gen_gan_loss + gen_mse_loss
            gen_loss.backward()
            gen_opt.step()

            # train discriminator
            fake_data = gen_out.detach()
            real_data = y.detach()
            nsamp = x.shape[0]
            fake_labels = torch.zeros(nsamp,1).cuda()
            real_labels = torch.ones(nsamp,1).cuda()

            data = torch.cat([real_data, fake_data], dim=0).detach()
            labels = torch.cat([real_labels, fake_labels], dim=0).detach()
            xx = torch.cat([x,x], dim=0).detach()

            discr_out = discr.forward(data, xx)
            discr_loss = nn.BCEWithLogitsLoss()(discr_out, labels)

            # mixup
            #beta = torch.from_numpy(np.random.beta(1,1,size=fake_data.shape[0]).astype(np.float32)).cuda()
            #mixup_data = fake_data * beta.reshape(-1,1,1) + real_data * (1 - beta.reshape(-1,1,1))
            #mixup_labels = fake_labels * beta.reshape(-1,1) + real_labels * (1 - beta.reshape(-1,1))
            #discr_out = discr.forward(mixup_data, x)

            # no mixup
            #mixup_data = torch.cat([fake_data, real_data], dim=0)
            #mixup_labels = torch.cat([fake_labels, real_labels], dim=0)
            #discr_out = discr.forward(mixup_data, torch.cat([x,x], dim=0))
            #fake_out = discr.forward(fake_data, x)
            #real_out = discr.forward(real_data, x)
            #discr_loss = nn.BCEWithLogitsLoss()(discr_out, mixup_labels).cuda()

            discr_loss.backward()
            discr_opt.step()

            discr_losses.append(detach_numpy(discr_loss).max())
            gen_mse_losses.append(detach_numpy(gen_mse_loss).max())
            gen_gan_losses.append(detach_numpy(gen_gan_loss).max())

            discr_losses = discr_losses[-1000:]
            gen_gan_losses = gen_gan_losses[-1000:]
            gen_mse_losses = gen_mse_losses[-1000:]

            if i % 1000 == 0:
                print(np.mean(discr_losses), np.mean(gen_gan_losses), np.mean(gen_mse_losses))
                plt.clf()
                plt.subplot(1,3,1)
                plt.imshow(detach_numpy(x[0]))
                plt.title("x")
                plt.subplot(1,3,2)
                plt.imshow(detach_numpy(y[0]))
                plt.title("y")
                plt.subplot(1,3,3)
                plt.imshow(detach_numpy(gen_out[0]))
                plt.title("o")
                plt.savefig("current_reconstr.png", dpi=250)
                
            if i % 10000 == 0:
                # get a sample.
                compr_sample_l = np.expand_dims(sample_pairs[0][0], 0)
                compr_sample_r = np.expand_dims(sample_pairs[0][2], 0)
                
                # do this on cpu so we dont run outta ram
                gen.eval()
                gen = gen.cpu()

                outs = []
                for compr_sample in [compr_sample_l, compr_sample_r]:
                    out = gen.forward(torch.from_numpy(compr_sample))
                    out = out.detach().cpu().numpy().squeeze()
                    out_i = get_istft(out)


                    outs.append(out_i.reshape(-1,1))
            
                outs = np.concatenate(outs, axis=1)
                outs /= np.max(np.abs(outs))

                # convert back into 16bit
                outs = (outs * 32767).astype(np.int16)
                wavfile.write("best_recon.wav", 44100, out_i.flatten())
            
                # move back to cuda
                gen = gen.cuda()

    
        except:
            print("probably an oom")
            break
    
    if len(loss_avg) == 0:
        continue

    loss_avg = np.mean(loss_avg)

    if loss_avg < best_loss:
        print("new best:", kernel_sz, lr, num_chan, loss_avg)

        compr_sample_l = np.expand_dims(sample_pairs[0][0], 0)
        compr_sample_r = np.expand_dims(sample_pairs[0][2], 0)
        
        # do this on cpu so we dont run outta ram
        gen.eval()
        gen = gen.cpu()

        outs = []
        for compr_sample in [compr_sample_l, compr_sample_r]:
            out = gen.forward(torch.from_numpy(compr_sample))
            out = out.detach().cpu().numpy().squeeze()
            out_i = get_istft(out)
            out_i /= np.max(np.abs(out_i))

            # convert back into 16bit
            out_i = (out_i * 32768).astype(np.int16)

            outs.append(out_i.reshape(-1,1))
    
        outs = np.concatenate(outs, axis=1)
        wavfile.write("best_recon.wav", 44100, out_i.flatten())
    
        # move back to cuda
        gen = gen.cuda()

        best_loss = loss_avg
        best = [gen, opt, datadim, nsteps]
    
best_model, opt, datadim, nsteps = best
sample_pairs = load_dataset(data_dir, datadim - 2)

loss_avg = []
i = 0
while True:
    i += 1

    x, y = make_batch(sample_pairs, datadim, nsteps=nsteps, cuda=True)

    best_model.train()

    opt.zero_grad()
    out = best_model.forward(x)
    loss = torch.mean(torch.abs(out - y))

    loss_np = loss.detach().cpu().numpy().flatten()[0]
    loss_avg.append(loss_np)
    loss_avg = loss_avg[-100:]

    if i % 100 == 0:
        for param_group in opt.param_groups:
            param_group['lr'] *= 0.999
            print("new lr: ", param_group['lr'])

        print(best_loss, np.mean(loss_avg), model_str)

    loss.backward()

    opt.step()
