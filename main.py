import matplotlib
matplotlib.use('TkAgg')  # or 'Agg' for non-display
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import butter, filtfilt

# Define signal parameters
Fs = 10000  # Sampling frequency (Hz)
T = 1  # Signal duration (seconds)
t = np.arange(0, T, 1/Fs)  # Time vector

# Create the original signal (sinusoidal)
freq_signal = 50  # Frequency of the original signal (Hz)
signal = np.sin(2 * np.pi * freq_signal * t)

# Create combined noise
# Sinusoidal noise with multiple frequencies
sinusoidal_noise = 0.4 * np.sin(2 * np.pi * 1000 * t) + \
                   0.3 * np.sin(2 * np.pi * 200 * t) + \
                   0.2 * np.sin(2 * np.pi * 1500 * t) + \
                   0.1 * np.sin(2 * np.pi * 3000 * t) + \
                   0.3 * np.sin(2 * np.pi * 48 * t) + \
                   0.3 * np.sin(2 * np.pi * 52 * t)  # Noise close to the original signal

# Gaussian noise
gaussian_noise = 0.3 * np.random.randn(len(t))

# Impulse noise
impulse_noise = np.zeros(len(t))
num_impulses = 20  # Number of impulses
impulse_indices = np.random.randint(0, len(t), num_impulses)
impulse_noise[impulse_indices] = 3 * (np.random.rand(num_impulses) - 0.5)

# White noise
white_noise = 0.2 * np.random.randn(len(t))

# Combine the noises
noise = sinusoidal_noise + gaussian_noise + impulse_noise + white_noise

# Combine the signal with noise
noisy_signal = signal + noise

# Design and apply a 4th-order bandpass filter
f_low = 40  # Lower passband frequency (Hz)
f_high = 60  # Upper passband frequency (Hz)

# Design the filter
b, a = butter(4, [f_low, f_high], fs=Fs, btype='bandpass')
filtered_signal_4 = filtfilt(b, a, noisy_signal)

# Frequency domain analysis of the noisy signal
N = len(noisy_signal)
fft_signal = np.fft.fft(noisy_signal)
f = np.fft.fftfreq(N, 1/Fs)  # Frequency vector
f = np.fft.fftshift(f)  # Shift frequencies to the positive range
magnitude = np.abs(np.fft.fftshift(fft_signal))  # Magnitude of the Fourier transform

# Frequency domain analysis of the filtered signal
fft_filtered_4 = np.fft.fft(filtered_signal_4)
magnitude_filtered_4 = np.abs(np.fft.fftshift(fft_filtered_4))  # Magnitude of the Fourier transform

# Display noisy signal in the time domain
plt.figure(figsize=(10, 6))
plt.subplot(3, 1, 1)
plt.plot(t, noisy_signal)
plt.title('Noisy Signal in Time Domain')
plt.xlabel('Time (s)')
plt.ylabel('Amplitude')
plt.grid(True)

# Display frequency domain analysis of noisy signal in logarithmic scale
plt.subplot(3, 1, 2)
plt.semilogy(f, magnitude)
plt.title('Noisy Signal in Frequency Domain (Log Scale)')
plt.xlabel('Frequency (Hz)')
plt.ylabel('Magnitude (Log scale)')
plt.grid(True)

# Display frequency domain analysis of noisy signal in linear scale
plt.subplot(3, 1, 3)
plt.plot(f, magnitude)
plt.title('Noisy Signal in Frequency Domain (Linear Scale)')
plt.xlabel('Frequency (Hz)')
plt.ylabel('Magnitude')
plt.grid(True)

# Display filtered signal in the time domain
plt.figure(figsize=(10, 6))
plt.plot(t, filtered_signal_4)
plt.title('Filtered Signal (Order 4) in Time Domain')
plt.xlabel('Time (s)')
plt.ylabel('Amplitude')
plt.grid(True)

# Display frequency domain analysis of filtered signal in logarithmic scale
plt.figure(figsize=(10, 6))
plt.semilogy(f, magnitude_filtered_4)
plt.title('Filtered Signal (Order 4) in Frequency Domain (Log Scale)')
plt.xlabel('Frequency (Hz)')
plt.ylabel('Magnitude (Log scale)')
plt.grid(True)

# Display frequency domain analysis of filtered signal in linear scale
plt.figure(figsize=(10, 6))
plt.plot(f, magnitude_filtered_4)
plt.title('Filtered Signal (Order 4) in Frequency Domain (Linear Scale)')
plt.xlabel('Frequency (Hz)')
plt.ylabel('Magnitude')
plt.grid(True)

plt.show()
