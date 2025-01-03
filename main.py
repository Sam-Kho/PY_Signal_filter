
import matplotlib
matplotlib.use('TkAgg')  # یا 'Agg' برای بدون نمایش
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import butter, filtfilt

# تعریف پارامترهای سیگنال
Fs = 10000  # فرکانس نمونه‌برداری (Hz)
T = 1  # مدت زمان سیگنال (ثانیه)
t = np.arange(0, T, 1/Fs)  # بردار زمان

# ایجاد سیگنال اصلی (sinusoidal)
freq_signal = 50  # فرکانس سیگنال اصلی (Hz)
signal = np.sin(2 * np.pi * freq_signal * t)

# ایجاد نویز ترکیبی
# نویز سینوسی با چندین فرکانس
sinusoidal_noise = 0.4 * np.sin(2 * np.pi * 1000 * t) + \
                   0.3 * np.sin(2 * np.pi * 200 * t) + \
                   0.2 * np.sin(2 * np.pi * 1500 * t) + \
                   0.1 * np.sin(2 * np.pi * 3000 * t) + \
                   0.3 * np.sin(2 * np.pi * 48 * t) + \
                   0.3 * np.sin(2 * np.pi * 52 * t)  # نویز نزدیک به سیگنال اصلی

# نویز گوسی (Gaussian Noise)
gaussian_noise = 0.3 * np.random.randn(len(t))

# نویز پالس (Impulse Noise)
impulse_noise = np.zeros(len(t))
num_impulses = 20  # تعداد پالس‌ها
impulse_indices = np.random.randint(0, len(t), num_impulses)
impulse_noise[impulse_indices] = 3 * (np.random.rand(num_impulses) - 0.5)

# نویز سفید (White Noise)
white_noise = 0.2 * np.random.randn(len(t))

# ترکیب نویزها
noise = sinusoidal_noise + gaussian_noise + impulse_noise + white_noise

# ترکیب سیگنال و نویز
noisy_signal = signal + noise

# طراحی و اعمال فیلتر میان‌گذر با مرتبه 4
f_low = 40  # فرکانس پایین باند عبور (Hz)
f_high = 60  # فرکانس بالای باند عبور (Hz)

# طراحی فیلتر
b, a = butter(4, [f_low, f_high], fs=Fs, btype='bandpass')
filtered_signal_4 = filtfilt(b, a, noisy_signal)

# تحلیل حوزه فرکانس سیگنال نویزی
N = len(noisy_signal)
fft_signal = np.fft.fft(noisy_signal)
f = np.fft.fftfreq(N, 1/Fs)  # بردار فرکانس
f = np.fft.fftshift(f)  # انتقال فرکانس‌ها به محدوده مثبت
magnitude = np.abs(np.fft.fftshift(fft_signal))  # قدر مطلق تبدیل فوریه

# تحلیل حوزه فرکانس سیگنال فیلترشده
fft_filtered_4 = np.fft.fft(filtered_signal_4)
magnitude_filtered_4 = np.abs(np.fft.fftshift(fft_filtered_4))  # قدر مطلق تبدیل فوریه

# نمایش سیگنال نویزی در حوزه زمان
plt.figure(figsize=(10, 6))
plt.subplot(3, 1, 1)
plt.plot(t, noisy_signal)
plt.title('Noisy Signal in Time Domain')
plt.xlabel('Time (s)')
plt.ylabel('Amplitude')
plt.grid(True)

# نمایش تحلیل فرکانسی سیگنال نویزی در مقیاس لگاریتمی
plt.subplot(3, 1, 2)
plt.semilogy(f, magnitude)
plt.title('Noisy Signal in Frequency Domain (Log Scale)')
plt.xlabel('Frequency (Hz)')
plt.ylabel('Magnitude (Log scale)')
plt.grid(True)

# نمایش تحلیل فرکانسی سیگنال نویزی در مقیاس خطی
plt.subplot(3, 1, 3)
plt.plot(f, magnitude)
plt.title('Noisy Signal in Frequency Domain (Linear Scale)')
plt.xlabel('Frequency (Hz)')
plt.ylabel('Magnitude')
plt.grid(True)

# نمایش سیگنال فیلترشده در حوزه زمان
plt.figure(figsize=(10, 6))
plt.plot(t, filtered_signal_4)
plt.title('Filtered Signal (Order 4) in Time Domain')
plt.xlabel('Time (s)')
plt.ylabel('Amplitude')
plt.grid(True)

# نمایش تحلیل فرکانسی سیگنال فیلترشده در مقیاس لگاریتمی
plt.figure(figsize=(10, 6))
plt.semilogy(f, magnitude_filtered_4)
plt.title('Filtered Signal (Order 4) in Frequency Domain (Log Scale)')
plt.xlabel('Frequency (Hz)')
plt.ylabel('Magnitude (Log scale)')
plt.grid(True)

# نمایش تحلیل فرکانسی سیگنال فیلترشده در مقیاس خطی
plt.figure(figsize=(10, 6))
plt.plot(f, magnitude_filtered_4)
plt.title('Filtered Signal (Order 4) in Frequency Domain (Linear Scale)')
plt.xlabel('Frequency (Hz)')
plt.ylabel('Magnitude')
plt.grid(True)

plt.show()
