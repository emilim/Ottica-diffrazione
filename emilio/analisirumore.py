import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import argparse
from scipy import stats

parser = argparse.ArgumentParser(description="Plot data da un file specificato")
parser.add_argument('filename', type=str, help='nome del file da leggere')

args = parser.parse_args()

# Lettura del file
df = pd.read_csv(args.filename, delimiter='\t', header=None)

# Creazione di un array numpy
array = df.values

passi_motore = array[:, 0]
intensity = array[:, 1]

plt.scatter(passi_motore, intensity)
plt.xlabel("Passi Motore")
plt.ylabel("Intensità")

plt.show()

media = np.mean(intensity)
deviazione_standard = np.std(intensity)
print(f"Media rumore: {media}", f"Deviazione standard rumore: {deviazione_standard/np.sqrt(len(intensity)-1)}", sep='\n')
# Histogram
counts, bins, _ = plt.hist(intensity, bins=20, edgecolor='black', alpha=0.7, range=(media - 50, media + 50), density=False)

# Bin width
bin_width = bins[1] - bins[0]

# Gaussian function
x = np.linspace(min(bins), max(bins), 1000)
gaussian = (1 / (deviazione_standard * np.sqrt(2 * np.pi))) * np.exp(-0.5 * ((x - media) / deviazione_standard) ** 2)
gaussian_scaled = gaussian * len(intensity) * bin_width  # Scale Gaussian to match histogram counts

# Plot Gaussian curve
plt.plot(x, gaussian_scaled, 'k', linewidth=2)

plt.xlabel("Intensità")
plt.ylabel("Frequenza")

plt.show()

    



