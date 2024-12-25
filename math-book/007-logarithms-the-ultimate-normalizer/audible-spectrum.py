import matplotlib.pyplot as plt
import numpy as np

# Define sound intensities in dB
sound_intensities_db = np.array([0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 110, 120, 130])
sound_descriptions = ['Threshold of hearing', 'Rustling leaves', 'Whisper', 'Quiet library', 'Bird calls', 'Conversation',
                      'Traffic', 'Vacuum cleaner', 'City traffic', 'Subway train', 'Concert', 'Sirens', 'Jet engine at takeoff', 'Threshold of pain']

# Create the plot
plt.figure(figsize=(10, 6))
plt.plot(sound_intensities_db, marker='o')
plt.title('Sound Intensity Spectrum')
plt.xlabel('Sound Events')
plt.ylabel('Decibels (dB)')
plt.xticks(ticks=np.arange(len(sound_descriptions)), labels=sound_descriptions, rotation=90)
plt.grid(True, which='both', linestyle='--', linewidth=0.5)

# Show plot
plt.tight_layout()
plt.show()
