import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import argparse
from scipy import stats
from scipy.stats import norm

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
minimo = -b/(2*a)

#metodo massima verosimiglianza
s_z=(2*a*passi_motore +b)*10/np.sqrt(12)
s_a1=np.sqrt(1/(np.sum(passi_motore**4/(s_z)**2)))
s_b1=np.sqrt(1/(np.sum(passi_motore**2/(s_z)**2)))
s_minimo1 = np.abs(minimo)*np.sqrt(s_b1**2/b**2 + s_a1**2/(4*a**2))
print(f"a={a} +- {s_a1}")
print(f"b={b} +- {s_b1}")
print(f"theta(1) max verosimiglianza= {minimo} +- {s_minimo1}")

plt.scatter(passi_motore, intensity)
#plt.scatter(minimo, polynomial(minimo), color="green")
#plt.errorbar(minimo, polynomial(minimo), xerr=s_minimo1)
#plt.plot(x_fit, y_fit, color="red")
plt.axhline(y=2281, color='b', linestyle='-', label='Rumore')
plt.yscale("log")
plt.xlabel("Passi Motore")
plt.ylabel("Intensity")
plt.title("Intensità vs Passi Motore")
plt.legend()
plt.show()

s_y= np.sqrt(1/np.size(intensity)*np.sum((intensity-polynomial(passi_motore))**2))
chi= np.sum((intensity-(a*passi_motore**2 + b*passi_motore + c))**2/s_y**2)

#print(f"s_y ={s_y} \n")
#print(f"chi={chi} , GDL= {np.size(intensity)-3}\n")

#conversione passi motore radiante
minimo_rad= minimo*0.0191 #mrad
s_conversione = 0.0191*np.sqrt((0.005/0.5)**2 + (0.5/(65.5))**2)
s_minimorad1=np.sqrt(np.pow(0.0191*s_minimo1, 2) + np.pow(minimo*s_conversione, 2))
print(f"minimo radianti verosimiglianza: {minimo_rad} +- {s_minimorad1} mrad \n")
print(f"s_conv={s_conversione}\n")



