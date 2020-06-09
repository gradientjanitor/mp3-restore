import torch
import random

import numpy as np

from torch import optim, nn

from matplotlib import pyplot as plt

# from gan import DiscrNet, ReconNet
import gan
from data_utils import get_stft, get_istft, load_dataset, make_batch

from scipy.io import wavfile

def detach_numpy(x):
    return x.detach().cpu().numpy()

def restore_sample(s, gen, filename="best_recon.wav"):
    print("restoring", s[-1])
    # get a sample.
    compr_sample_l = np.expand_dims(s[0], 0)
    compr_sample_r = np.expand_dims(s[2], 0)
    
    # do this on cpu so we dont run outta vram
    gen.eval()
    gen = gen.cpu()

    outs = []
    for compr_sample in [compr_sample_l, compr_sample_r]:
        compr_sample_mag = compr_sample[0:1,:compr_sample.shape[1]//2,:]
        compr_sample_phase = compr_sample[0:1,compr_sample.shape[1]//2:,:]

        out_mag = gen.forward(torch.from_numpy(compr_sample_mag))
        out_mag = out_mag.detach().cpu().numpy().squeeze()

        # get phase from compr data
        out = np.concatenate([out_mag, compr_sample_phase.squeeze()], axis=0)

        out_i = get_istft(out)
        outs.append(out_i.reshape(-1,1))

    outs = np.concatenate(outs, axis=1)
    outs /= np.max(np.abs(outs))

    # convert back into 16bit
    outs = (outs * 32767).astype(np.int16)
    wavfile.write(filename, 44100, outs)

    # move back to cuda
    gen = gen.cuda()

def plot_reconstruction(x, y, gen_out, filename="current_reconstr.png"):
    plt.clf()
    plt.subplot(1,3,1)
    plt.imshow(detach_numpy(x))
    plt.title("low bitrate")
    plt.subplot(1,3,2)
    plt.imshow(detach_numpy(y))
    plt.title("high bitrate")
    plt.subplot(1,3,3)
    plt.imshow(detach_numpy(gen_out))
    plt.title("reconstruction")
    plt.savefig(filename, dpi=250)

data_dir = "../data/raw"

# grab some random hyperparams for the model
fft_len = random.choice([256,512])
kernel_sz = random.choice([3,5,7,9,11])
lr = random.choice([0.0001])
nsteps = random.choice([32,48,64])
batch_size = random.choice([8,16,32])
datadim = (fft_len + 2) // 2
residual = True #random.choice([True,False])
num_chan_proto = random.choice([[1,2,3,4,5]])
num_chan = np.asarray(num_chan_proto) * datadim

print("generating stfts...")
sample_pairs = load_dataset(data_dir, fft_len)
x, y = make_batch(sample_pairs, 2, datadim, nsteps=nsteps, cuda=True)
print("done.")

from importlib import reload
reload(gan)
gen = gan.ReconNet(datadim, kernel_sz=kernel_sz, num_chan=num_chan, residual=residual).cuda()
discr = gan.DiscrNet(x.cpu(), datadim, kernel_sz=9, num_chan=[32,64,128,192,256]).cuda()
gen_opt = optim.Adam(gen.parameters(), lr=lr)
discr_opt = optim.Adam(discr.parameters(), lr=lr)

loss_avg, discr_losses, discr_accs, discr_gen_accs, gen_gan_losses, gen_mse_losses = [], [], [], [], [], []

i = 0
while True:
    i += 1

    # make sure models are in training mode (so dropout/batchnorm/etc all work properly)
    gen.train()
    discr.train()

    # ye olde optimizer grad clear
    gen_opt.zero_grad()
    discr_opt.zero_grad()

    # train discriminator
    # get fresh data.  mmm
    x, y = make_batch(sample_pairs, batch_size, datadim, nsteps=nsteps, cuda=True)
    gen_out = gen.forward(x)

    fake_data = gen_out.detach()
    real_data = y.detach()
    nsamp = x.shape[0]
    real_labels = torch.zeros(nsamp,1).cuda()
    fake_labels = torch.ones(nsamp,1).cuda()

    data = torch.cat([real_data, fake_data], dim=0).detach()
    labels = torch.cat([real_labels, fake_labels], dim=0).detach()

    discr_out = discr.forward(data)
    # discr_loss = -(torch.mean(discr_out[:discr_out.shape[0]//2]) - torch.mean(discr_out[discr_out.shape[0]//2:]))
    discr_loss = nn.BCEWithLogitsLoss()(discr_out, labels)
    discr_acc = np.mean((detach_numpy(discr_out) > 0) == (detach_numpy(labels) > 0.5))

    discr_loss.backward()
    discr_opt.step()
    
    # train generator
    x, y = make_batch(sample_pairs, batch_size, datadim, nsteps=nsteps, cuda=True)
    gen_out = gen.forward(x)
    discr_gen_out = discr.forward(gen_out)
    discr_gen_acc = np.mean(detach_numpy(discr_gen_out) > 0)

    # cross entropy for GAN + l1 loss for reconstruction
    gen_gan_loss = nn.BCEWithLogitsLoss()(discr_gen_out, torch.zeros(gen_out.shape[0], 1).cuda())
    # gen_gan_loss = -torch.mean(discr_gen_out)
    gen_mse_loss = torch.mean(torch.pow(gen_out - y, 2))
    gen_loss = gen_gan_loss + gen_mse_loss
    gen_loss.backward()
    gen_opt.step()

    discr_losses.append(detach_numpy(discr_loss).max())
    gen_mse_losses.append(detach_numpy(gen_mse_loss).max())
    gen_gan_losses.append(detach_numpy(gen_gan_loss).max())
    discr_accs.append(discr_acc)
    discr_gen_accs.append(discr_gen_acc)

    discr_losses = discr_losses[-100:]
    discr_accs = discr_accs[-100:]
    discr_gen_accs = discr_gen_accs[-100:]
    gen_gan_losses = gen_gan_losses[-100:]
    gen_mse_losses = gen_mse_losses[-100:]

    # every once in a while, spit out a spectrogram of the low-bitrate mp3, the orig, and the
    # reconstruction
    if i % 100 == 0:
        print(gen_out.max(), gen_out.min())
        print(np.mean(discr_accs), np.mean(discr_gen_accs), np.mean(discr_losses), np.mean(gen_gan_losses), np.mean(gen_mse_losses))
        plot_reconstruction(x[0], y[0], gen_out[0])
        
    # every once in a while, shove a full MP3 throught the model and let's listen to it
    if i % 10000 == 0:
        s = random.choice(range(len(sample_pairs)))
        restore_sample(sample_pairs[s], gen)
