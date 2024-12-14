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

plt.scatter(passi_motore, intensity)
plt.xlabel("Passi Motore")
plt.ylabel("Intensity")
plt.title("Intensità vs Passi Motore")
plt.show()

mask_x1 = (passi_motore < 20) #| (passi_motore > 83)  # Maschera per i dati fuori dall'intervallo
x_filtered1 = passi_motore[mask_x1]
y_filtered1 = intensity[mask_x1]

mask_x2 = (passi_motore > 60)
x_filtered2 = passi_motore[mask_x2]
y_filtered2 = intensity[mask_x2]
# Fit parabolico (polinomio di grado 1)
coefficients1, cov_matrix1 = np.polyfit(x_filtered1, y_filtered1, 1, full=False, cov=True)
polynomial1 = np.poly1d(coefficients1)

coefficients2, cov_matrix2 = np.polyfit(x_filtered2, y_filtered2, 1, full=False, cov=True)
polynomial2 = np.poly1d(coefficients2)


# Visualizza il fit e i dati
x_fit1 = np.linspace(min(passi_motore), max(passi_motore), 100)
y_fit1 = polynomial1(x_fit1)
x_fit2 = np.linspace(min(passi_motore), max(passi_motore), 100)
y_fit2 = polynomial2(x_fit2)

# Coefficienti delle due rette
a1, b1 = coefficients1
a2, b2 = coefficients2

s_a1 = np.sqrt(cov_matrix1[0, 0])  # Incertezza su a1
s_b1 = np.sqrt(cov_matrix1[1, 1])  # Incertezza su b1

s_a2 = np.sqrt(cov_matrix2[0, 0])  # Incertezza su a2
s_b2 = np.sqrt(cov_matrix2[1, 1])  # Incertezza su b2

# Trova l'intersezione
if a1 != a2:  # Assicurati che le due rette non siano parallele
    x_intersection = (b2 - b1) / (a1 - a2)
    y_intersection = a1 * x_intersection + b1  # Puoi usare anche l'altra equazione per calcolare y
    s_x = np.sqrt((s_b1**2+s_b2**2)/(a1-a2)**2 + (b2-b1)**2*((s_a1**2+s_a2**2)/(a1-a2)**4))

print(f"Punto di intersezione: x = {x_intersection} +- {s_x}, y = {y_intersection}")

plt.scatter(x_intersection, y_intersection, color="green")
plt.scatter(passi_motore, intensity, label="Dati originali", color="blue")
plt.plot(x_fit1, y_fit1, label="Fit lineare", color="red")
plt.plot(x_fit2, y_fit2, label="Fit lineare", color="red")
plt.xlabel("x")
plt.ylabel("y")
plt.legend()
plt.show()

# Sovrascrivere completamente il file di testo con i nuovi risultati
#with open('risultati.txt', 'w') as file:
#    file.write(f"theta zero, 1 fenditura intervallo -200 a 235:\n")
#    file.write(f"theta_zero = {x_intersection} +- {s_x} passi motore\n")


    



