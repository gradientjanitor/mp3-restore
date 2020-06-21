# Using GANs to restore low-bitrate MP3s
## Examples:
Sound samples can be found in "low_bitrate_example.mp3" and "reconstructed_example.mp3".  Not too shabby!

## Usage:
Copy a bunch of good-sounding mp3's into data/mp3_high

Run the convert.py tool to re-encode them into garbage-sounding 64kbps MP3s

Run restorer/train.py to train a GAN that tries to reconstruct portions of the low-bitrate MP3s in the frequency domain.

Here's an example reconstruction of the spectrogram:
![GitHub Logo](example_reconstruction.png)

# Details
Writeup coming soon...
