import torch
import random

import numpy as np

from torch import optim, nn

from matplotlib import pyplot as plt

from gan import DiscrNet, ReconNet
from data_utils import get_stft, get_istft, load_dataset, make_batch

def detach_numpy(x):
    return x.detach().cpu().numpy()

data_dir = "../data/raw"

# grab some random hyperparams for the model
fft_len = random.choice([256,512])
kernel_sz = random.choice([3,5,7,9,11])
lr = random.choice([0.0001])
nsteps = random.choice([32,48,64])
batch_size = random.choice([8,16,32])
datadim = fft_len + 2
add_noise = random.choice([True,False])
residual = random.choice([True,False])
num_chan_proto = random.choice([[1,2,3],[1,2,3,2,1],[1,2,3,4,5]])
num_chan = np.asarray(num_chan_proto) * datadim

print("generating stfts...")
sample_pairs = load_dataset(data_dir, fft_len)
x, y = make_batch(sample_pairs, 2, datadim, nsteps=nsteps, cuda=True)
print("done.")

gen = ReconNet(datadim, kernel_sz=kernel_sz, num_chan=num_chan, add_noise=add_noise, residual=residual).cuda()
discr = DiscrNet(x.cpu(), datadim, kernel_sz=9, num_chan=[512,256,128,64]).cuda()
gen_opt = optim.Adam(gen.parameters(), lr=lr)
discr_opt = optim.Adam(discr.parameters(), lr=lr)

loss_avg = []

discr_losses = []
gen_gan_losses= []
gen_mae_losses= []

i = 0
while True:
    i += 1
    x, y = make_batch(sample_pairs, batch_size, datadim, nsteps=nsteps, cuda=True)

    # make sure models are in training mode (so dropout/batchnorm/etc all work properly)
    gen.train()
    discr.train()

    # ye olde optimizer grad clear
    gen_opt.zero_grad()
    discr_opt.zero_grad()

    # train generator
    gen_out = gen.forward(x)
    discr_gen_out = discr.forward(gen_out, x)

    # cross entropy for GAN + l1 loss for reconstruction
    gen_gan_loss = nn.BCEWithLogitsLoss()(discr_gen_out, torch.ones(gen_out.shape[0], 1).cuda())
    gen_mae_loss = torch.mean(torch.abs(gen_out - y))
    gen_loss = gen_gan_loss + gen_mae_loss
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

    discr_loss.backward()
    discr_opt.step()

    discr_losses.append(detach_numpy(discr_loss).max())
    gen_mae_losses.append(detach_numpy(gen_mae_loss).max())
    gen_gan_losses.append(detach_numpy(gen_gan_loss).max())

    discr_losses = discr_losses[-1000:]
    gen_gan_losses = gen_gan_losses[-1000:]
    gen_mae_losses = gen_mae_losses[-1000:]

    # every once in a while, spit out a spectrogram of the low-bitrate mp3, the orig, and the
    # reconstruction
    if i % 1000 == 0:
        print(np.mean(discr_losses), np.mean(gen_gan_losses), np.mean(gen_mae_losses))
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
        
    # every once in a while, shove a full MP3 throught the model and let's listen to it
    if i % 10000 == 0:
        # get a sample.
        compr_sample_l = np.expand_dims(sample_pairs[0][0], 0)
        compr_sample_r = np.expand_dims(sample_pairs[0][2], 0)
        
        # do this on cpu so we dont run outta vram
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
