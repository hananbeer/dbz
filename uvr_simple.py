from __future__ import annotations

import os
import gc
import time
import torch
import librosa
import platform
import warnings
import numpy as np
import soundfile as sf
import onnxruntime as ort

from onnx import load
from onnx2pytorch import ConvertModel

warnings.filterwarnings("ignore")

BASE_PATH = os.path.dirname(os.path.abspath(__file__))
# Change the current working directory to the base path
os.chdir(BASE_PATH)

if platform.system() == "Darwin":
    is_windows = False
    is_macos = True
    application_extension = ".dmg"
elif platform.system() == "Linux":
    is_windows = False
    is_macos = False
    application_extension = ".zip"
elif platform.system() == "Windows":
    is_windows = True
    is_macos = False
    application_extension = ".exe"

MODELS_DIR = os.path.join(BASE_PATH, 'models')

def clear_gpu_cache():
    gc.collect()
    if is_macos:
        torch.mps.empty_cache()
    else:
        torch.cuda.empty_cache()


def normalize(wave, is_normalize=False):
    """Normalize audio"""

    maxv = np.abs(wave).max()
    if maxv > 1.0:
        if is_normalize:
            print("Above clipping threshold.")
            wave /= maxv
    
    return wave

def export_audio(file_path: str, wave, samplerate=44100):
    wave = normalize(wave, is_normalize=False)
    sf.write(file_path, wave, samplerate) #, subtype=wav_type_set)

class STFT:
    def __init__(self, n_fft, hop_length, dim_f, device):
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.window = torch.hann_window(window_length=self.n_fft, periodic=True)
        self.dim_f = dim_f
        self.device = device

    def __call__(self, x):
        
        x_is_mps = not x.device.type in ["cuda", "cpu"]
        if x_is_mps:
            x = x.cpu()

        window = self.window.to(x.device)
        batch_dims = x.shape[:-2]
        c, t = x.shape[-2:]
        x = x.reshape([-1, t])
        x = torch.stft(x, n_fft=self.n_fft, hop_length=self.hop_length, window=window, center=True,return_complex=False)
        x = x.permute([0, 3, 1, 2])
        x = x.reshape([*batch_dims, c, 2, -1, x.shape[-1]]).reshape([*batch_dims, c * 2, -1, x.shape[-1]])

        if x_is_mps:
            x = x.to(self.device)

        return x[..., :self.dim_f, :]

    def inverse(self, x):
        
        x_is_mps = not x.device.type in ["cuda", "cpu"]
        if x_is_mps:
            x = x.cpu()

        window = self.window.to(x.device)
        batch_dims = x.shape[:-3]
        c, f, t = x.shape[-3:]
        n = self.n_fft // 2 + 1
        f_pad = torch.zeros([*batch_dims, c, n - f, t]).to(x.device)
        x = torch.cat([x, f_pad], -2)
        x = x.reshape([*batch_dims, c // 2, 2, n, t]).reshape([-1, 2, n, t])
        x = x.permute([0, 2, 3, 1])
        x = x[..., 0] + x[..., 1] * 1.j
        x = torch.istft(x, n_fft=self.n_fft, hop_length=self.hop_length, window=window, center=True)
        x = x.reshape([*batch_dims, 2, -1])

        if x_is_mps:
            x = x.to(self.device)

        return x

class SeperateMDX:
    def __init__(self,
                 audio_file,
                 export_path):
        self.model = None
        self.model_path = os.path.join(MODELS_DIR, 'UVR-MDX-NET-Inst_HQ_3.onnx')
        self.use_gpu = True

        self.progress_value = 0
        self.audio_file = audio_file
        self.export_path = export_path

        self.mdx_batch_size = 1
        # self.overlap = 0.25
        self.overlap_mdx = 'Default'

        # taken from the model_data.json
        #self.max_dim_f_set = 3072
        #self.max_dim_t_set = 8 # 2**8
        #self.mdx_n_fft_scale_set = 6144
        
        self.adjust = 1
        #self.dim_c = 4
        self.hop = 1024

        self.dim_f = 3072
        self.dim_t = 2 ** 8
        self.mdx_stem_count = 1
        self.compensate = 1.022
        self.n_fft = 6144
        self.samplerate = 44100
        self.mdx_segment_size = 1024

        # self.use_opencl = False

        if is_macos and torch.backends.mps.is_available():
            self.device = 'mps'
            self.is_other_gpu = True
        elif torch.cuda.is_available(): # and not self.use_opencl:
            self.device = 'cuda'
            self.run_type = ['CUDAExecutionProvider']
            self.is_other_gpu = False
        else:
            self.device = 'cpu'
            self.run_type = ['CPUExecutionProvider']
            self.is_other_gpu = False

    def set_progress(self, value):
        self.progress_value = value
        print('%.2f%%' % (self.progress_value * 100))

    def load_model(self):
        # TODO: optimize this? 1024 seems to be faster, even though it needs to ConvertModel()
        if self.mdx_segment_size == self.dim_t and not self.is_other_gpu:
            ort_ = ort.InferenceSession(self.model_path, providers=self.run_type)
            self.model = lambda spek:ort_.run(None, {'input': spek.cpu().numpy()})[0]
        else:
            self.model = ConvertModel(load(self.model_path))
            self.model.to(self.device).eval()


    def seperate(self):
        self.set_progress(0)

        if not self.model:
            self.load_model()

        # load waveform
        mix = prepare_mix(self.audio_file)

        assert mix.shape[0] == 2, 'Stereo audio is required'

        # run inference
        source = self.demix(mix)
        
        # save output
        filename = os.path.basename(self.audio_file)
        output_path = os.path.join(self.export_path, f'{filename[:-4]}_Instrumental.wav')

        export_audio(output_path, source.T, samplerate=self.samplerate)
        self.set_progress(1)
    
        clear_gpu_cache()

    def initialize(self):
        # self.n_bins = self.n_fft // 2 + 1
        self.trim = self.n_fft // 2
        self.chunk_size = self.hop * (self.mdx_segment_size - 1)
        self.gen_size = self.chunk_size - 2 * self.trim
        self.stft = STFT(self.n_fft, self.hop, self.dim_f, self.device)

    def demix_inner(self, mix):
        pass

    def demix(self, mix):
        self.initialize()
        
        total_waves = []
        chunk_size = self.chunk_size
        overlap = self.overlap_mdx
        if overlap == 'Default':
            step = self.chunk_size - self.n_fft 
        else:
            step = int((1 - overlap) * chunk_size)

        gen_size = chunk_size - 2 * self.trim
        pad_size = gen_size + self.trim - ((mix.shape[-1]) % gen_size)
        padding = np.zeros((2, pad_size), dtype='float32')
        trim_padding = np.zeros((2, self.trim), dtype='float32')
        padded_mix = np.concatenate((trim_padding, mix, padding), 1)
        channel_count, sample_count = padded_mix.shape

        result = np.zeros((1, 2, sample_count), dtype=np.float32)
        divider = np.zeros((1, 2, sample_count), dtype=np.float32)
        total = 0
        total_chunks = (sample_count + step - 1) // step

        for start in range(0, sample_count, step):
            total += 1
            end = min(start + chunk_size, sample_count)

            current_chunk_size = end - start

            if overlap == 0:
                window = None
            else:
                window = np.hanning(current_chunk_size)
                window = np.tile(window[None, None, :], (1, 2, 1))

            mix_part_ = padded_mix[:, start:end]
            if end != start + chunk_size:
                pad_size = (start + chunk_size) - end
                mix_part_ = np.concatenate((mix_part_, np.zeros((2, pad_size), dtype='float32')), axis=-1)

            mix_part = torch.tensor([mix_part_], dtype=torch.float32).to(self.device)
            mix_waves = mix_part.split(self.mdx_batch_size)
            
            with torch.no_grad():
                for mix_wave in mix_waves:
                    self.set_progress(total / total_chunks)

                    tar_waves = self.run_model(mix_wave)
                    
                    if window is not None:
                        tar_waves[..., :current_chunk_size] *= window 
                        divider[..., start:end] += window
                    else:
                        divider[..., start:end] += 1

                    result[..., start:end] += tar_waves[..., :end-start]
            
        tar_waves = result / divider
        total_waves.append(tar_waves)
        total_waves = np.vstack(total_waves)[:, :, self.trim:-self.trim]
        tar_waves = np.concatenate(total_waves, axis=-1)[:, :mix.shape[-1]]
        
        source = tar_waves[:,0:None]

        # if not is_match_mix:
        source *= self.compensate

        return source

    def run_model(self, mix):
        spek = self.stft(mix.to(self.device)) * self.adjust
        spek[:, :, :3, :] *= 0 
        spec_pred = self.model(spek)
        return self.stft.inverse(torch.tensor(spec_pred).to(self.device)).cpu().detach().numpy()

def prepare_mix(mix):
    if not isinstance(mix, np.ndarray):
        mix, sr = librosa.load(mix, mono=False, sr=44100)
    else:
        mix = mix.T

    if mix.ndim == 1:
        mix = np.asfortranarray([mix,mix])

    return mix

def process_simple():
    audio_file = r'C:/Users/J/Downloads/Eminem - Rap God (Explicit) - EminemVEVO (youtube).mp3'
    export_path = './debug_simple'

    stime = time.perf_counter()
    time_elapsed = lambda:f'Time Elapsed: {time.strftime("%H:%M:%S", time.gmtime(int(time.perf_counter() - stime)))}'

    if not os.path.isdir(export_path):
        os.makedirs(export_path) 

    seperator = SeperateMDX(audio_file, export_path)
    seperator.seperate()

    clear_gpu_cache()

    print('done')
    print(time_elapsed())



if __name__ == "__main__":
    process_simple()
