try:
    import librosa
except:
    print('missing required module librosa, install with:\n\tpip install librosa==0.9.2')
    exit(1)

import os
import time
import numpy as np
import argparse
import soundfile

ap = argparse.ArgumentParser()
ap.add_argument('input', type=str)
ap.add_argument('output', type=str, nargs='?', default='output.wav')
ap.add_argument('--gpu', type=bool, default=True)

# TODO: make args
volume_music = 1.0
volume_vocals = 0.0

args = ap.parse_args()

input_path = os.path.abspath(args.input)
output_path = os.path.abspath(args.output)
output_dir = os.path.dirname(output_path)

if not os.path.isdir(output_dir):
    os.makedirs(output_dir) 

# load model
print('loading model...', end=' ')

stime = time.perf_counter()
show_time = lambda: print('%.2fs' % (time.perf_counter() - stime))

import zen_demixer
separator = zen_demixer.ZenDemixer(use_gpu=args.gpu)
show_time()

# load audio
print('loading audio...', end=' ')
stime = time.perf_counter()

mix, samplerate = librosa.load(input_path, mono=False, sr=None)
if mix.ndim == 1:
    mix = np.asfortranarray([mix, mix])
 
show_time()

# run inference
print('demixing........', end='\r')
stime = time.perf_counter()
source = separator.demix(mix, progress_cb=lambda p: print('demixing........ %.2f%%' % (100.*p), end='\r'), volume_music=volume_music, volume_vocals=volume_vocals)
print('demixing........', end='     ')
show_time()

# save output
print('saving output...', end=' ')
stime = time.perf_counter() 
soundfile.write(output_path, source.T, samplerate=samplerate)
show_time()
