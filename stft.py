import numpy as np
import torch
import scipy.signal
import matplotlib.pyplot as plt

adjust = 1
hop = 1024
dim_f = 3072
dim_t = 256
n_fft = 6144
mdx_segment_size = 256
chunk_size = hop * (mdx_segment_size - 1)

input_buffer = np.loadtxt('./input/input_buffer.txt')

class STFT:
    def __init__(self, n_fft, hop, dim_f, device='cpu'):
        self.n_fft = n_fft
        self.hop = hop
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
        x = torch.stft(x, n_fft=self.n_fft, hop_length=self.hop, window=window, center=True, return_complex=True)
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
        x = torch.istft(x, n_fft=self.n_fft, hop_length=self.hop, window=window, center=True)
        x = x.reshape([*batch_dims, 2, -1])

        if x_is_mps:
            x = x.to(self.device)

        return x

def stft_og(x, n_fft, hop, dim_f):
    return STFT(n_fft, hop, dim_f)(torch.tensor(x)).cpu().numpy()

def istft_og(x, n_fft, hop, dim_f):
    return STFT(n_fft, hop, dim_f).inverse(torch.tensor(x)).cpu().numpy()

def stft_np(x, n_fft, hop, dim_f, window='hann'):
    # Compute the number of frames
    frame_count = (x.shape[1] // hop) + 1

    # Create the window function
    if window == 'hann':
        # NOTE: the torch impl also has periodic=True:
        # torch.hann_window(window_length=self.n_fft, periodic=True)
        win = np.hanning(n_fft)
    elif window == 'hamming':
        win = np.hamming(n_fft)
    else:
        win = np.ones(n_fft)  # Rectangular window
    
    # Initialize the STFT matrix
    pad_to = (frame_count-1) * hop + n_fft
    padh = np.zeros((x.shape[0], (pad_to - x.shape[1]) // 2))
    x = np.concatenate([padh, x, padh], axis=1)

    result = np.zeros((2, 2, n_fft // 2 + 1, frame_count), dtype=np.float32)

    # Compute STFT
    for i in range(frame_count):
        # Extract frame
        frame = x[:, i * hop : i * hop + n_fft]
        
        # TODO: instead of this signal should be padded in both sides
        if frame.shape[1] != win.shape[0]:
            break

        # Apply window function
        windowed_frame = frame * win
        
        # Compute FFT
        res = np.fft.fft(windowed_frame)
        res2 = np.array([[res[0].real, res[0].imag], [res[1].real, res[1].imag]])
        result[..., i] = res2[..., :n_fft // 2 + 1]

    # can make batch dim by adding (1, ..) and [:, ...]
    return result.reshape((4, result.shape[2], result.shape[3]))[:, :dim_f, :]

def stft_sc(x, n_fft, hop, dim_f, window='hann'):
    f, t, zXX = scipy.signal.stft(x, nfft=n_fft, nperseg=hop, window=window)
    return zXX[:, :dim_f, :]

def istft_np(x, n_frame, n_hop, dim_f, window='hann'):
    freq_bins, frame_count = x.shape[1:]

    # stft_matrix_complex0 = stft_matrix[0] + 1j * stft_matrix[1]
    # stft_matrix_complex1 = stft_matrix[2] + 1j * stft_matrix[3]
    # stft_matrix = np.array([stft_matrix_complex0, stft_matrix_complex1])

    x = x[:2, ...]

    # Create the window function
    if window == 'hann':
        win = np.hanning(n_frame)
    elif window == 'hamming':
        win = np.hamming(n_frame)
    else:
        win = np.ones(n_frame)

    # win = win + 1j * win
    win = np.array([win, win])
    
    # Calculate the length of the output signal
    output_length = n_hop * (frame_count - 1) + n_frame
    
    # Initialize the output array
    output = np.zeros((2, output_length), dtype=np.float32)
    
    # Initialize normalization array
    norm = np.zeros((2, output_length), dtype=np.float32)
    
    # Reconstruct the signal
    for i in range(frame_count):
        frame = np.fft.ifft(x[:, :, i])
        windowed_frame = frame * win[:, :freq_bins]
        output[:, i * n_hop : i * n_hop + freq_bins] += windowed_frame.real
        norm[:, i * n_hop : i * n_hop + n_frame] += win ** 2
    
    # Normalize the output
    # output /= np.where(norm > 1e-10, norm, 1)
    
    return output


def istft_sc(x, n_fft, n_hop, dim_f, window='hann', noverlap=None):
    if noverlap is None:
        noverlap = n_hop // 2

    t, zXX = scipy.signal.istft(x, nfft=n_fft, nperseg=n_hop, window=window)#, input_onesided=True,
    
    return zXX

x_og = stft_og(input_buffer, n_fft, hop, dim_f)
print('x_og.shape', x_og.shape)

x_np = stft_np(input_buffer, n_fft, hop, dim_f)
print('x_np.shape', x_np.shape)
# # print(np.abs(x_og - x_np).max())

x_sc = stft_sc(input_buffer, n_fft, hop, dim_f)
print('x_sc.shape', x_sc.shape)

# f, t, x_sc = scipy.signal.stft(x, n_fft, hop, window='hann', axis=2)
# f[:dim_f]
# t[:dim_t]
# torch.view_as_real(x_sc)


### PLOTTING ###

# x_og2 = x_og.reshape(2, 2, x_og.shape[1], x_og.shape[2])
# x_np2 = x_np.reshape(2, 2, x_np.shape[1], x_np.shape[2])
# channel0 = x_og2[0]

# # Compute the magnitude of the STFT
# stft_mag0 = np.abs(x_og2[0])[0]
# # stft_mag1 = np.abs(x_og2[1])[1]
# stft_mag1 = np.abs(np.abs(x_og2[0])[0] - np.abs(x_np2[0])[0])

# # Plot the spectrogram
# plt.figure(figsize=(20, 8))
# plt.subplot(1, 2, 1)
# plt.imshow(stft_mag0[:100], aspect='auto', origin='lower', cmap='inferno', vmax=100)
# plt.title('Original Spectrogram')

# plt.subplot(1, 2, 2)
# plt.imshow(stft_mag1[:100], aspect='auto', origin='lower', cmap='inferno', vmax=100)
# plt.colorbar(format='%+2.0f dB')
# plt.title('Subtracted Spectrogram')
# plt.xlabel('Time')
# plt.ylabel('Frequency')
# plt.show()

# original for testing
# wave_og = istft_og(x_og, n_fft, hop, dim_f)

# np -> np similarly to og -> np has pitch and volume changes
# probably due to broken normalization, could be broken padding?
# wave_np = istft_np(x_og, n_fft, hop, dim_f)

# sc -> sc works, but og -> sc doesn't
wave_sc = istft_sc(x_sc, n_fft, hop, dim_f)

import soundfile
# soundfile.write('wave_og.wav', wave_og[0], 44100)
# soundfile.write('wave_np.wav', wave_np[0].real, 44100)
soundfile.write('wave_sc.wav', wave_sc[0].real, 44100)
exit(0)

# Plot input_buffer as an audio waveform
plt.figure(figsize=(20, 8))
plt.subplot(1, 2, 1)
plt.plot(input_buffer[0][:4096])
plt.title('Input Audio Waveform')
plt.xlabel('Time')
plt.ylabel('Amplitude')


plt.subplot(1, 2, 2)
plt.plot(wave_np[0][:4096])
plt.title('Reconstructed Audio Waveform')
plt.xlabel('Time')
plt.ylabel('Amplitude')

plt.show()

exit(0)