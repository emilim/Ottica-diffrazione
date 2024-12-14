import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import argparse
from scipy import stats

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

coefficients, cov_matrix = np.polyfit(passi_motore, intensity, 2, full=False, cov=True)
polynomial = np.poly1d(coefficients)

x_fit=np.linspace(np.min(passi_motore), np.max(passi_motore), 100)
y_fit=polynomial(x_fit)

a, b, c= coefficients
s_a =np.sqrt(cov_matrix[0, 0])
s_b =np.sqrt(cov_matrix[1, 1])
minimo = -b/(2*a)
s_minimo = minimo*np.sqrt(s_b**2/b**2 + s_a**2/4*a**2)
print(f"theta(-1)= {minimo} +- {s_minimo}")
print(f"a={a} +- {s_a}")
print(f"b={b} +- {s_b}")

plt.scatter(passi_motore, intensity)
plt.scatter(minimo, polynomial(minimo), color="green")
plt.errorbar(minimo, polynomial(minimo), xerr=np.abs(s_minimo))
plt.plot(x_fit, y_fit, color="red")
plt.xlabel("Passi Motore")
plt.ylabel("Intensity")
plt.title("Intensità vs Passi Motore")
plt.show()

