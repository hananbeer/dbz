import os
import gc
import time
import onnx
import torch
import numpy as np
import librosa
import platform
import warnings
import soundfile
import onnxruntime

from onnx2pytorch import ConvertModel

warnings.filterwarnings("ignore", "(The given NumPy array is not writable|PySoundFile failed|To copy construct from a tensor)")

if platform.system() == "Darwin":
    is_windows = False
    is_macos = True
elif platform.system() == "Linux":
    is_windows = False
    is_macos = False
elif platform.system() == "Windows":
    is_windows = True
    is_macos = False

def clear_gpu_cache():
    gc.collect()
    if is_macos:
        torch.backends.mps.empty_cache()
    else:
        torch.cuda.empty_cache()

def stft(signal, n_fft, hop_length):
    # Create a Hann window
    window = np.hanning(n_fft)
    # Calculate the number of frames
    num_frames = 1 + (len(signal) - n_fft) // hop_length
    # Pad the signal
    padded_signal = np.pad(signal, (0, n_fft), mode='constant')
    # Initialize the STFT matrix
    stft_matrix = np.zeros((num_frames, n_fft), dtype=complex)

    for i in range(num_frames):
        start = i * hop_length
        frame = padded_signal[start:start + n_fft] * window
        stft_matrix[i, :] = np.fft.fft(frame)

    return stft_matrix

class STFT:
    def __init__(self, n_fft, hop_length, dim_f, device):
        self.n_fft = n_fft
        self.hop_length = hop_length
        #self.window = torch.hann_window(window_length=self.n_fft, periodic=True)
        # using numpy instead: (does not result in exact output, but like 0.1% difference)
        self.window = torch.tensor(np.hanning(self.n_fft).astype(np.float32))
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
        x = torch.stft(x, n_fft=self.n_fft, hop_length=self.hop_length, window=window, center=True, return_complex=True)
        x = torch.view_as_real(x)
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

class ZenDemixer:
    model_path = './models/UVR-MDX-NET-Inst_HQ_3.onnx'

    adjust = 1
    hop = 1024
    dim_f = 3072
    dim_t = 2 ** 8
    mdx_stem_count = 1
    compensate = 1.022
    n_fft = 6144
    mdx_segment_size = 1024

    # samplerate = 44100
    buffer_ms = None # 1500

    def __init__(self, use_gpu=True):
        if is_macos and use_gpu and torch.backends.mps.is_available():
            self.device = 'mps'
        elif use_gpu and torch.cuda.is_available():
            self.device = 'cuda'
        else:
            self.device = 'cpu'

        # taken from the model_data.json
        #self.max_dim_f_set = 3072
        #self.max_dim_t_set = 8 # 2**8
        #self.mdx_n_fft_scale_set = 6144
        
        self.chunk_size = self.hop * (self.mdx_segment_size - 1)
        self.stft = STFT(self.n_fft, self.hop, self.dim_f, self.device)
        self._load_model()

    def _load_model(self):
        print('loading model')

        # TODO: optimize this? 1024 seems to be faster, even though it needs to ConvertModel()
        if self.mdx_segment_size == self.dim_t and self.device != 'mps':
            provider = 'CUDAExecutionProvider' if self.device == 'cuda' else 'CPUExecutionProvider'
            inference = onnxruntime.InferenceSession(self.model_path, providers=[provider])
            self.model = lambda spek: inference.run(None, {'input': spek.cpu().numpy()})[0]
        else:
            self.model = ConvertModel(onnx.load(self.model_path))
            self.model.to(self.device).eval()

    def run_model(self, mix, adjust=1.0):
        """
        expects `mix` to be a `torch.tensor` of shape (batch_size=1, channels=2, samples=self.chunk_size)
        (batch sizes larger than 1 are not supported, model was trained on channels=2 and chunk_size=(1024*1023))
        """
        spec = self.stft(mix.to(self.device))
        if adjust != 1.0:
            spec *= adjust

        spec[:, :, :3, :] *= 0 
        with torch.no_grad():
            spec_pred = self.model(spec)

        tensor = torch.tensor(spec_pred).to(self.device)
        return self.stft.inverse(tensor).cpu().detach().numpy()

    def demix(self, mix, buffer_size=None):
        """
        the model always processes chunks of self.chunk_size samples

        buffer_size can be used to process in smaller chunks by zero padding the buffer
        to self.chunk_size, hence buffer_size must be less than or equal to self.chunk_size
        """
        chunk_size = self.chunk_size
        channel_count, sample_count = mix.shape
        assert sample_count > 0, 'no samples'
        assert channel_count == 2, 'stereo audio is required'

        # pad the mix to the nearest chunk size
        # this makes the logic much simpler because sample_count is divisible by chunk_size
        if sample_count < chunk_size:
            pad_size = chunk_size - sample_count
        else:
            pad_size = chunk_size - (sample_count % chunk_size)

        if pad_size > 0:
            sample_count += pad_size
            padding = np.zeros((channel_count, pad_size), dtype=np.float32)
            mix = np.concatenate((mix, padding), axis=-1)

        result = np.zeros((channel_count, sample_count), dtype=np.float32)
        if not buffer_size:
            buffer_size = chunk_size
        else:
            assert buffer_size <= chunk_size, 'buffer_size must be less than or equal to chunk_size'

        for start in range(0, sample_count, buffer_size):
            stime = time.perf_counter()

            end = start + buffer_size
            chunk = mix[:, start:end]
            if chunk_size != buffer_size:
                padding = np.zeros((channel_count, chunk_size - buffer_size), dtype=np.float32)
                chunk = np.concatenate((chunk, padding), axis=-1)

            chunk_tensor = torch.tensor(np.array([chunk]), dtype=torch.float32).to(self.device)
            spec_pred = self.run_model(chunk_tensor, adjust=self.adjust)
            result[:, start:end] = spec_pred[..., :buffer_size]

            print('%.2f%% (%.2fs)' % (100. * end / sample_count, time.perf_counter() - stime))

        return result[:, :sample_count - pad_size]

    def demix_file(self, input_path, output_path):
        # load waveform
        # TODO: pass samplerate here?
        print('loading waveform')
        mix, samplerate = librosa.load(input_path, mono=False, sr=None)
        if mix.ndim == 1:
            mix = np.asfortranarray([mix, mix])

        # run inference
        print('demixing')
        source = self.demix(mix)

        # save output
        print('saving output')
        soundfile.write(output_path, source.T, samplerate=samplerate)

        clear_gpu_cache()

def process_simple(input_path, output_path):
    seperator = ZenDemixer(use_gpu=True)
    seperator.demix_file(input_path, output_path)


if __name__ == "__main__":
    input_path = r'./input/Eminem - Rap God.mp3'
    # input_path = r'./input/Why iii Love The Moon.mp4'
    # input_path = r'./input_buffer.wav'
    output_dir = './output'

    filename = os.path.basename(input_path)
    output_path = os.path.join(output_dir, f'{filename[:-4]}_Instrumental.wav')

    if not os.path.isdir(output_dir):
        os.makedirs(output_dir) 

    stime = time.perf_counter()

    process_simple(input_path, output_path)

    print('done: %.2fs' % (time.perf_counter() - stime))

