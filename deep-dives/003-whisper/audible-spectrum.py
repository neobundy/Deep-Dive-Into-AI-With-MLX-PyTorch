import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd

# Data preparation
data = pd.DataFrame({
    'Frequency (Hz)': [20, 20000],
    'Audibility': [1, 1]  # Constant value as audibility isn't changing
})

# Create the plot
plt.figure(figsize=(10, 6))
audible_spectrum_plot = sns.lineplot(x='Frequency (Hz)', y='Audibility', data=data)

# Setting the plot title and labels
plt.title('Audible Spectrum from 20Hz to 20kHz')
plt.xlabel('Frequency (Hz)')
plt.ylabel('Audibility')
plt.xticks([20, 1000, 5000, 10000, 15000, 20000], ['20Hz', '1kHz', '5kHz', '10kHz', '15kHz', '20kHz'])  # Custom x-ticks

# Show the plot
plt.show()
