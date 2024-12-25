from scipy.io import wavfile
import matplotlib.pyplot as plt

# Read the WAV file
file_path = './data/hello.wav'
sample_rate, data = wavfile.read(file_path)

# Generate time axis
time = [float(n) / sample_rate for n in range(len(data))]

# Plot the data
plt.figure(figsize=(12, 6))
plt.plot(time, data)
plt.title('Audio Waveform')
plt.xlabel('Time (seconds)')
plt.ylabel('Amplitude')
plt.show()
