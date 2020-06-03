# tool to take in some "high quality" MP3's, re-encode them
# to a garbage bitrate, then save both as WAV files because
# it's easier to open in python

import glob
import subprocess
import sys
import os

for f in glob.glob("mp3_low/*"):
    os.remove(f)

for f in glob.glob("raw/*"):
    os.remove(f)

for f in glob.glob("mp3_high/*"):
    high_mp3 = f
    low_mp3 = f.replace("mp3_high", "mp3_low")
    high_raw = f.replace("mp3_high", "raw").replace(".mp3", "_high.wav")
    low_raw = f.replace("mp3_high", "raw").replace(".mp3", "_low.wav")

    subprocess.call('ffmpeg -i "%s" -map 0:a:0 -b:a 64k "%s"' % (high_mp3, low_mp3), shell=True)
    subprocess.call('ffmpeg -i "%s" "%s"' % (high_mp3, high_raw), shell=True)
    subprocess.call('ffmpeg -i "%s" "%s"' % (low_mp3, low_raw), shell=True)
