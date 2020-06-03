# Using GANs to restore low-bitrate MP3s
## Usage:
Copy a bunch of good-sounding mp3's into data/mp3_high
Run the convert.py tool to re-encode them into garbage-sounding 64kbps MP3s

Run freq/conv_train.py to train a GAN that tries to reconstruct portions of the low-bitrate MP3s in the ~~frequency domain~~.

Currently, it sounds pretty okay after training for a while, but some MP3 artifacting remains.

This is research-grade code (meaning: little to no documentation, doesn't do much outside of printing out accuracy numbers and showing pretty plots of frequency-domain reconstructions.) at the present, but once I've nailed down a GAN architecture that works pretty well, it'll have features like:
- Saving models and reconstructing MP3s in a given directory
