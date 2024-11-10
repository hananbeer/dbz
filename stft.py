import numpy as np
import torch
import librosa
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

chunk = input_buffer[:, :chunk_size]
print('chunk.shape', chunk.shape)

### TORCH IMPL ###
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

### NUMPY IMPL ###
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

def istft_np(x, n_frame, n_hop, dim_f, window='hann'):
    freq_bins, frame_count = x.shape[1:]

    # stft_matrix_complex0 = stft_matrix[0] + 1j * stft_matrix[1]
    # stft_matrix_complex1 = stft_matrix[2] + 1j * stft_matrix[3]
    # stft_matrix = np.array([stft_matrix_complex0, stft_matrix_complex1])

    x = np.array([x[0] + 1j * x[1], x[2] + 1j * x[3]])

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
    output /= np.where(norm > 1e-10, norm, 1)
    
    return output

### SCIPY.SIGNAL IMPL ###
one_sided = True
boundary = ['even', 'odd', 'constant', 'zeros', None]
def stft_sc(x, n_fft, hop, dim_f, window='hann'):
    # window = np.hanning(hop)
    window = scipy.signal.get_window('hann', hop)
    f, t, zXX = scipy.signal.stft(x, nfft=n_fft, nperseg=hop, window=window, return_onesided=one_sided, boundary=None, padded=False)
    complex = zXX[:, :dim_f, :]
    quad_dim = np.array([complex[0].real, complex[0].imag, complex[1].real, complex[1].imag])
    scale = np.sqrt(1.0 / window.sum() ** 2)
    res = quad_dim / scale
    return res

def istft_sc(x, n_fft, n_hop, dim_f, window='hann', noverlap=None):
    # if noverlap is None:
    #     noverlap = n_hop // 2

    x = np.array([x[0] + 1j * x[1], x[2] + 1j * x[3]])
    t, zXX = scipy.signal.istft(x, nfft=n_fft, nperseg=n_hop, window=window, input_onesided=one_sided)
    
    return zXX

### LIBROSA IMPL ###
def stft_lr(x, n_fft, hop, dim_f, window='hann'):
    zXX = librosa.stft(x, n_fft=n_fft, hop_length=hop, window=window, center=True)
    complex = zXX[:, :dim_f, :]
    quad_dim = np.array([complex[0].real, complex[0].imag, complex[1].real, complex[1].imag])
    return quad_dim

def istft_lr(x, n_frame, n_hop, dim_f, window='hann'):
    res = librosa.istft(x, n_fft=n_fft, hop_length=n_hop, window=window)
    return res

### WAVE TO SPEC ###

# stft_name = 'np'
# stft_name = 'sc'
stft_name = 'lr'

stft_impl = globals()['stft_' + stft_name]
istft_impl = globals()['istft_' + stft_name]

spec_torch = stft_og(chunk, n_fft, hop, dim_f)
print('spec_torch.shape', spec_torch.shape)

# idk why hop * 2
# hop_sc = hop * 2
spec_new = stft_impl(chunk, n_fft, hop, dim_f)
print('spec_new.shape', spec_new.shape)

### SPEC PLOTTING ###

plt.figure(figsize=(20, 16))
width = 1 # 4
for i in range(width):
    plt.subplot(2, width, i+1)
    plt.imshow(spec_torch[i, :100], aspect='auto', origin='lower', cmap='inferno')#, vmin=-10, vmax=100)
    plt.colorbar(format='%+2.0f dB')
    plt.title(f'Original Spectrogram {i}')

    plt.subplot(2, width, width + i+1)
    plt.imshow(spec_new[i, :100], aspect='auto', origin='lower', cmap='inferno')#, vmin=-10, vmax=100)
    plt.colorbar(format='%+2.0f dB')
    plt.title(f'Custom Impl Spectrogram {i}')
    plt.xlabel('Time')
    plt.ylabel('Frequency')
plt.show()

### SPEC TO WAVE ###

# original for testing
wave_torch = istft_og(spec_new, n_fft, hop, dim_f)

# np -> np similarly to og -> np has pitch and volume changes
# probably due to broken normalization, could be broken padding?
wave_new = istft_impl(spec_new, n_fft, hop, dim_f)

import soundfile
# soundfile.write('wave_torch.wav', wave_torch[0], 44100)
soundfile.write(f'wave_{stft_name}.wav', wave_new[1], 48000)
# exit(0)


### WAVE PLOTTING ###

# Plot input_buffer as an audio waveform
plt.figure(figsize=(20, 8))
plt.subplot(1, 2, 1)
plt.plot(chunk[0][:4096])
plt.title('Input Audio Waveform')
plt.xlabel('Time')
plt.ylabel('Amplitude')


plt.subplot(1, 2, 2)
plt.plot(wave_new[0][:4096])
plt.title('Reconstructed Audio Waveform')
plt.xlabel('Time')
plt.ylabel('Amplitude')

plt.show()

exit(0)