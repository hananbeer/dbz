import torch
import numpy as np
import warnings
import onnxruntime
import scipy.signal
# import librosa
import librosa_stft as librosa

warnings.filterwarnings("ignore", "(The given NumPy array is not writable|PySoundFile failed|To copy construct from a tensor)")

# python zen_from_file.py "input\Eminem - Rap God.mp3" --nogpu

class STFT_np:
    def __init__(self, n_fft, hop_length, dim_f):
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.window = np.hanning(self.n_fft).astype(np.float32)
        self.dim_f = dim_f

    def __call__(self, x):
        # Assuming x is a numpy array
        batch_dims = x.shape[:-2]
        c, t = x.shape[-2:]
        x = x.reshape([-1, t])
        # Simulate STFT using numpy (this is a placeholder for actual STFT implementation)
        # This is a simplified example and does not perform actual STFT
        x_stft = np.fft.fft(x, n=self.n_fft)[..., :self.dim_f]
        x_stft = np.abs(x_stft)
        x_stft = x_stft.reshape([*batch_dims, c, -1, self.dim_f])
        return x_stft.astype(np.float32)

    def inverse(self, x):
        # Assuming x is a numpy array
        batch_dims = x.shape[:-3]
        c, f, t = x.shape[-3:]
        n = self.n_fft // 2 + 1
        f_pad = np.zeros([*batch_dims, c, n - f, t])
        x = np.pad(x, ((0, 0), (0, 0), (0, n - f), (0, 0)), 'constant')
        # Simulate ISTFT using numpy (this is a placeholder for actual ISTFT implementation)
        # This is a simplified example and does not perform actual ISTFT
        x_istft = np.fft.ifft(x, n=self.n_fft)
        x_istft = x_istft.reshape([*batch_dims, 2, -1])
        return x_istft.astype(np.float32)


def stft(x, n_frame, n_hop, window='hann'):
    """
    Compute the Short-Time Fourier Transform (STFT) of a signal.
    
    Parameters:
    x (array): Input signal
    n_frame (int): Frame length
    n_hop (int): Hop length between frames
    window (str): Window function (default: 'hann')
    
    Returns:
    stft (2D array): Complex STFT matrix
    """
    x = x.cpu().numpy()

    # Compute the number of frames
    frame_count = 7 + (x.shape[1] - n_frame) // n_hop
    
    # Create the window function
    if window == 'hann':
        win = np.hanning(n_frame)
    elif window == 'hamming':
        win = np.hamming(n_frame)
    else:
        win = np.ones(n_frame)  # Rectangular window
    
    # Initialize the STFT matrix
    result = np.zeros((2, 2, n_frame // 2 + 1, frame_count), dtype=np.float32)
    
    try:
        # Compute STFT
        for i in range(frame_count):
            # Extract frame
            frame = x[:, i * n_hop : i * n_hop + n_frame]
            
            # Apply window function
            windowed_frame = frame * win
            
            # Compute FFT
            res = np.fft.fft(windowed_frame)
            res2 = np.array([[res[0].real, res[0].imag], [res[1].real, res[1].imag]])
            result[..., i] = res2[..., :n_frame // 2 + 1]
    except:
        print('stft')

    return result.reshape((1, 4, result.shape[2], result.shape[3]))[:, :, :n_frame // 2, :]

def istft(stft_matrix, n_frame, n_hop, window='hann'):
    frame_count, freq_bins = stft_matrix.shape
    
    # Create the window function
    if window == 'hann':
        win = np.hanning(n_frame)
    elif window == 'hamming':
        win = np.hamming(n_frame)
    else:
        win = np.ones(n_frame)
    
    # Calculate the length of the output signal
    output_length = n_hop * (frame_count - 1) + n_frame
    
    # Initialize the output array
    output = np.zeros(output_length)
    
    # Initialize normalization array
    norm = np.zeros(output_length)
    
    # Reconstruct the signal
    for i in range(frame_count):
        frame = np.fft.ifft(stft_matrix[i, :])
        windowed_frame = frame * win
        output[i * n_hop : i * n_hop + n_frame] += windowed_frame
        norm[i * n_hop : i * n_hop + n_frame] += win ** 2
    
    # Normalize the output
    output /= np.where(norm > 1e-10, norm, 1)
    
    return output



class STFT:
    def __init__(self, n_fft, hop_length, dim_f, device='cpu'):
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


one_sided = True
boundary = ['even', 'odd', 'constant', 'zeros', None]
def stft_sc(x, n_fft, hop, dim_f, window='hann'):
    # print('x.shape', x.shape)
    hop *= 2
    # window = np.hanning(hop)
    f, t, zXX = scipy.signal.stft(x, nfft=n_fft, nperseg=hop, window=window, return_onesided=one_sided, boundary=boundary[0], padded=True)
    complex = zXX[:, :dim_f, :256]
    res = np.array([complex[0].real, complex[0].imag, complex[1].real, complex[1].imag]).reshape((1, 4, complex.shape[1], complex.shape[2]))
    # print('res.shape', res.shape)
    return res

def istft_sc(x, n_fft, hop, dim_f, window='hann', noverlap=None):
    # if noverlap is None:
    #     noverlap = n_hop // 2
    hop *= 2
    x = x[0]
    x = np.array([x[0] + 1j * x[1], x[2] + 1j * x[3]])
    t, zXX = scipy.signal.istft(x, nfft=n_fft, nperseg=hop, window=window, input_onesided=one_sided)
    
    return zXX


### LIBROSA IMPL ###
def stft_lr(x, n_fft, hop, dim_f, window='hann'):
    zXX = librosa.stft(x, n_fft=n_fft, hop_length=hop, window=window, center=True)
    complex = zXX[:, :dim_f, :]
    quad_dim = np.array([complex[0].real, complex[0].imag, complex[1].real, complex[1].imag])
    return np.array([quad_dim])

def istft_lr(x, n_fft, n_hop, dim_f, window='hann'):
    res = librosa.istft(x, n_fft=n_fft, hop_length=n_hop, window=window, dtype=np.float32)
    res2 = res[:, ::2, :]
    # res2 = res
    # print(res2.shape)
    return res2


class ZenDemixer:
    # NOTE: quantization did not help much
    # this worked better than the convert.py script: (still not as good as the original onnx model)
    # python -m onnxruntime.quantization.preprocess --input=./models\UVR-MDX-NET-Inst_HQ_3.onnx --output=./models/demixer_m.onnx
    model_path = './models/UVR-MDX-NET-Inst_HQ_3.onnx'

    adjust = 1
    hop = 1024
    dim_f = 3072
    dim_t = 2 ** 8
    n_fft = 6144
    mdx_segment_size = 256
    chunk_size = hop * (mdx_segment_size - 1)

    def __init__(self, use_gpu=True):
        if use_gpu and torch.backends.mps.is_available():
            self.device = 'mps'
        elif use_gpu and torch.cuda.is_available():
            self.device = 'cuda'
        else:
            self.device = 'cpu'

        # TODO: should keep stft in cpu? seems just as fast or faster
        self.stft = STFT(self.n_fft, self.hop, self.dim_f) # self.device)
        self._load_model()

    def _load_model(self):
        # onnxruntime.set_default_logger_severity(0)  # Verbose logging

        provider = 'CUDAExecutionProvider' if self.device == 'cuda' else 'CPUExecutionProvider'
        inference = onnxruntime.InferenceSession(self.model_path, providers=[provider])
        self.model = lambda spec: inference.run(None, {'input': spec})[0]
        # exit(0)

    def run_model(self, mix, adjust=1.0, inverse=False):
        """
        expects `mix` to be a `torch.tensor` of shape (batch_size=1, channels=2, samples=self.chunk_size)
        (batch sizes larger than 1 are not supported, model was trained on channels=2 and chunk_size=(1024*1023))
        """
        # print('0', mix.shape)
        spec = stft_lr(mix[0].cpu().numpy(), self.n_fft, self.hop, self.dim_f)
        # spec = stft(mix.to(self.device)[0], self.n_fft, self.hop)
        # spec = self.stft(mix.to(self.device))
        # print('1', spec.shape)
        if adjust != 1.0:
            spec *= adjust

        # TODO: seems to not be necessary?
        spec[:, :, :3, :] *= 0
        # NOTE: onnx supports batch_size > 1, demo:
        # spec = torch.cat([spec, spec], dim=0)
        with torch.no_grad():
            spec_pred = self.model(spec)#[:1, ...]
        # print('2', spec_pred.shape)
        tensor = torch.tensor(spec_pred).to(self.device)
        # print('3', tensor.shape)
        if inverse:
            tensor = spec - tensor
        return istft_lr(tensor.cpu().numpy(), self.n_fft, self.hop, self.dim_f)
        # return self.stft.inverse(tensor).cpu().detach().numpy()

    def demix(self, mix, buffer_size=None, progress_cb=None, inverse=False):
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
            chunk = mix[:, start:start + buffer_size]
            end = start + chunk.shape[1]
            if chunk.shape[1] < chunk_size:
                padding = np.zeros((channel_count, chunk_size - chunk.shape[1]), dtype=np.float32)
                chunk = np.concatenate((chunk, padding), axis=-1)

            chunk_tensor = torch.tensor(np.array([chunk]), dtype=torch.float32).to(self.device)
            spec_pred = self.run_model(chunk_tensor, adjust=self.adjust, inverse=inverse)
            result[:, start:end] = spec_pred[..., :buffer_size]

            if progress_cb:
                progress_cb(end / sample_count)

        return result[:, :sample_count - pad_size]
