import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import argparse

parser = argparse.ArgumentParser(description="Plot data da un file specificato")
parser.add_argument('filename', type=str, help='nome del file da leggere')

parser.add_argument('--start', type=int, default=None, help='Passo motore iniziale (opzionale)')
parser.add_argument('--end', type=int, default=None, help='Passo motore finale (opzionale)')


args = parser.parse_args()

# Lettura del file
df = pd.read_csv(args.filename, delimiter='\t', header=None)

# Creazione di un array numpy
array = df.values

passi_motore = array[:, 0]
intensity = array[:, 1]

# Applica il filtro sull'intervallo se specificato
if args.start is not None or args.end is not None:
    mask = np.ones(len(passi_motore), dtype=bool)
    if args.start is not None:
        mask &= passi_motore >= args.start
    if args.end is not None:
        mask &= passi_motore <= args.end
    passi_motore = passi_motore[mask]
    intensity = intensity[mask]

plt.plot(passi_motore, intensity)
plt.xlabel("Passi Motore")
plt.ylabel("Intensity")
plt.title("Intensità vs Passi Motore")
plt.show()
