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

mask_x1 = (passi_motore < -11) #| (passi_motore > 83)  # Maschera per i dati fuori dall'intervallo
x_filtered1 = passi_motore[mask_x1]
y_filtered1 = intensity[mask_x1]

mask_x2 = (passi_motore > 83)
x_filtered2 = passi_motore[mask_x2]
y_filtered2 = intensity[mask_x2]
# Fit parabolico (polinomio di grado 1)
coefficients1, cov_matrix1 = np.polyfit(x_filtered1, y_filtered1, 1, full=False, cov=True)
polynomial1 = np.poly1d(coefficients1)

coefficients2, cov_matrix2 = np.polyfit(x_filtered2, y_filtered2, 1, full=False, cov=True)
polynomial2 = np.poly1d(coefficients2)


# Visualizza il fit e i dati
x_fit1 = np.linspace(-120, 61, 100)
y_fit1 = polynomial1(x_fit1)
x_fit2 = np.linspace(0, 170, 100)
y_fit2 = polynomial2(x_fit2)

# Coefficienti delle due rette
a1, b1 = coefficients1
a2, b2 = coefficients2

#s_a1 = np.sqrt(cov_matrix1[0, 0])  # Incertezza su a1
#s_b1 = np.sqrt(cov_matrix1[1, 1])  # Incertezza su b1

#s_a2 = np.sqrt(cov_matrix2[0, 0])  # Incertezza su a2
#s_b2 = np.sqrt(cov_matrix2[1, 1])  # Incertezza su b2

#incertezze parametri massima verosimiglianza
s_z1=a1*10/np.sqrt(12) #a1*s_x   s_x=deltapassi/sqrt(12)
s_a1=np.sqrt(1/(np.sum(x_filtered1**2/(s_z1**2))))
s_z2=a2*10/np.sqrt(12)
s_a2=np.sqrt(1/(np.sum(x_filtered2**2/(s_z2**2))))
print(f"s_a1={s_a1} e s_a2={s_a2}")

s_b1=np.sqrt(1/(np.abs((np.sum(x_filtered1/(s_z1**2))))))
s_b2=np.sqrt(1/(np.abs((np.sum(x_filtered2/(s_z2**2))))))
print(f"s_b1={s_b1} e s_b2={s_b2}")

# Trova l'intersezione
if a1 != a2:  # Assicurati che le due rette non siano parallele
    print(f"rette non parallele")
    x_intersection = (b2 - b1) / (a1 - a2)
    y_intersection = a1 * x_intersection + b1 
    s_x = np.sqrt((s_b1**2+s_b2**2)/((a1-a2)**2) + (b2-b1)**2*((s_a1**2+s_a2**2)/((a1-a2)**4)))


print(f"Punto di intersezione: x = {x_intersection} +- {s_x} passi motore, y = {y_intersection}")

plt.scatter(x_intersection, y_intersection, color="green", label="theta 0 con incertezza")
plt.errorbar(x_intersection, y_intersection, xerr=s_x, color="green")
plt.scatter(passi_motore, intensity, label="Dati originali", color="blue")
plt.plot(x_fit1, y_fit1, label="Fit lineare", color="red")
plt.plot(x_fit2, y_fit2, color="red")
plt.xlabel("x")
plt.ylabel("y")
plt.legend()
plt.show()

#conversione passi motore radiante
theta0_rad= x_intersection*0.0191 #mrad
s_conversione = 0.0191*np.sqrt((0.005/0.5)**2 + (0.5/65.5)**2)
s_theta0=np.sqrt(np.pow(0.0191*s_x, 2) + np.pow(x_intersection*s_conversione, 2))
print(f"theta_0={theta0_rad} +-{s_theta0} mrad")

#chi-quadro
chi1=np.sum((y_filtered1-a1*x_filtered1-b1)**2/s_z1**2)   
chi2=np.sum((y_filtered2-a2*x_filtered2-b2)**2/s_z2**2)  

print(f"chi1={chi1}, GDL {np.size(x_filtered1)} \n")
print(f"chi2={chi2}, GDL {np.size(x_filtered2)} \n")



