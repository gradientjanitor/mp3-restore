import glob
import time

from scipy.io import wavfile
from scipy.signal import stft, istft
import torch
import random

import numpy as np

M = 15
B = 12

def get_stft(x, fft_len):
    xf = stft(x.flatten(), nperseg=fft_len, return_onesided=True)[2]

    xm = (np.log(1e-10+np.abs(xf)) + B) / M
    xp = np.angle(xf)
    xp /= np.pi

    print(xm.min(), xm.max())

    xmp = np.concatenate([xm, xp], axis=0)

    return xmp


def get_istft(x):
    fft_dim = x.shape[0] // 2
    xm = np.exp(M * x[:fft_dim,:] - B)
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

        # disregard if the sample is less than 30sec, or if it's mono
        if orig_sample.shape[0] < 44100*30 or len(orig_sample.shape) != 2:
            continue
        
        # load, at most, a 30sec clip from a song.  we dont want to be stft'ing all day.
        print(orig_sample.shape)
        offset = random.choice(range(orig_sample.shape[0] - 44100 * 30))
        orig_sample = orig_sample[offset:offset + 44100 * 30,:]
        compr_sample = compr_sample[offset:offset + 44100 * 30,:]

        compr_stft_l = get_stft(compr_sample[:,0:1], fft_len)
        orig_stft_l = get_stft(orig_sample[:,0:1], fft_len)
        compr_stft_r = get_stft(compr_sample[:,1:2], fft_len)
        orig_stft_r = get_stft(orig_sample[:,1:2], fft_len)

        sample_pairs.append([compr_stft_l, orig_stft_l, compr_stft_r, orig_stft_r, f])

    return sample_pairs


def make_batch(sample_pairs, batch_size, datadim, nsteps=64, cuda=False):
    x = np.zeros((batch_size, datadim, nsteps), dtype=np.float32)
    y = np.zeros((batch_size, datadim, nsteps), dtype=np.float32)

    sample_idxs = np.random.permutation(len(sample_pairs))[:batch_size]

    for i, sample_idx in enumerate(sample_idxs):
        sample = sample_pairs[sample_idx]

        compr_sample_l, orig_sample_l, compr_sample_r, orig_sample_r, filename = sample

        # choose random channel
        if np.random.rand() < 0.5:
            compr_sample, orig_sample = compr_sample_l, orig_sample_l
        else:
            compr_sample, orig_sample = compr_sample_r, orig_sample_r

        sstart = np.random.randint(0, compr_sample.shape[1] - nsteps - 1)
        compr_sample = compr_sample[:,sstart:sstart+nsteps]
        orig_sample = orig_sample[:,sstart:sstart+nsteps]

        x[i,:,:] = compr_sample[:compr_sample.shape[0] // 2,:]
        y[i,:,:] = orig_sample[:orig_sample.shape[0] // 2,:]

    x, y = torch.from_numpy(x), torch.from_numpy(y)
    if cuda:
        x, y = x.cuda(), y.cuda()

    return x, y
