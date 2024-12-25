import torch
import seaborn as sns
import matplotlib.pyplot as plt

# Generate a random electric guitar signal with extreme values
torch.manual_seed(42)  # For reproducibility
guitar_signal = torch.randn(100) * 20  # Simulate random extreme values for signal amplitude

# A conceptual compressor for normalization using log normalization
normalized_signal = torch.sign(guitar_signal) * torch.log1p(torch.abs(guitar_signal))

# Plotting the signals before and after normalization using seaborn
plt.figure(figsize=(15, 6))

# Original signal plot
plt.subplot(1, 2, 1)
sns.lineplot(data=guitar_signal.numpy(), color="blue", label="Original Signal")
plt.title("Original Guitar Signal")
plt.xlabel("Sample")
plt.ylabel("Amplitude")

# Normalized signal plot
plt.subplot(1, 2, 2)
sns.lineplot(data=normalized_signal.numpy(), color="red", label="Normalized Signal")
plt.title("Normalized Guitar Signal")
plt.xlabel("Sample")
plt.ylabel("Normalized Amplitude")

plt.tight_layout()
plt.show()
