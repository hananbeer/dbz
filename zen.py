import onnx
import torch
import numpy as np
import warnings
import onnxruntime

from onnx2pytorch import ConvertModel

warnings.filterwarnings("ignore", "(The given NumPy array is not writable|PySoundFile failed|To copy construct from a tensor)")

class STFT:
    def __init__(self, n_fft, hop_length, dim_f, device):
        self.n_fft = n_fft
        self.hop_length = hop_length
        # self.window = torch.hann_window(window_length=self.n_fft, periodic=True)
        self.window = torch.tensor(np.hanning(self.n_fft).astype(np.float32)).to(device)
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
        self.stft = STFT(self.n_fft, self.hop, self.dim_f, 'cpu') # self.device)
        self._load_model()

    def _load_model(self):
        provider = 'CUDAExecutionProvider' if self.device == 'cuda' else 'CPUExecutionProvider'
        inference = onnxruntime.InferenceSession(self.model_path, providers=[provider])
        self.model = lambda spec: inference.run(None, {'input': spec.cpu().numpy()})[0]

    def run_model(self, mix, adjust=1.0, inverse=False):
        """
        expects `mix` to be a `torch.tensor` of shape (batch_size=1, channels=2, samples=self.chunk_size)
        (batch sizes larger than 1 are not supported, model was trained on channels=2 and chunk_size=(1024*1023))
        """
        spec = self.stft(mix.to(self.device))
        if adjust != 1.0:
            spec *= adjust

        # spec[:, :, :3, :] *= 0
        # NOTE: onnx supports batch_size > 1, demo:
        # spec = torch.cat([spec, spec], dim=0)
        with torch.no_grad():
            spec_pred = self.model(spec)#[:1, ...]

        tensor = torch.tensor(spec_pred).to(self.device)
        if inverse:
            tensor = spec - tensor
        return self.stft.inverse(tensor).cpu().detach().numpy()

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
