import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.fft import fft, fftfreq

# Parameters
fs_line = 50
f_main = 100
fs = 5000
T = 1.0
N = int(fs*T)
t = np.linspace(0, T, N, endpoint=False)

# Fault frequencies
f1, f2 = 300, 450

# Signal
signal = (
    1.0*np.sin(2*np.pi*f_main*t) +
    0.3*np.sin(2*np.pi*f1*t) +
    0.2*np.sin(2*np.pi*f2*t)
)

# Save dataset
pd.DataFrame({"Time(s)": t, "Vibration": signal}) \
  .to_csv("bearing_fault.csv", index=False)

# FFT
fft_vals = (2.0/N)*np.abs(fft(signal))
freqs = fftfreq(N, 1/fs)

# Plot
plt.figure(figsize=(6,4))
plt.plot(freqs[:N//2], fft_vals[:N//2], linewidth=1.5)
plt.xlim(0,600); plt.ylim(0,1.1)
plt.xlabel("Frequency (Hz)")
plt.ylabel("Normalized Amplitude")
plt.title("Bearing Fault (synthetic Vibration) – FFT Spectrum")
plt.grid(True, linestyle="--", alpha=0.5)
plt.tight_layout()
plt.savefig("bearing_fault_fft.png", dpi=300)
plt.show()
