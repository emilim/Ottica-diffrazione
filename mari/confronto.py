import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import argparse

# Creazione del parser per gli argomenti
parser = argparse.ArgumentParser(description="Plot data da due file specificati")
parser.add_argument('filename1', type=str, help='Nome del primo file da leggere')
parser.add_argument('filename2', type=str, help='Nome del secondo file da leggere')

parser.add_argument('--start1', type=int, default=None, help='Passo motore iniziale per il primo file (opzionale)')
parser.add_argument('--end1', type=int, default=None, help='Passo motore finale per il primo file (opzionale)')

parser.add_argument('--start2', type=int, default=None, help='Passo motore iniziale per il secondo file (opzionale)')
parser.add_argument('--end2', type=int, default=None, help='Passo motore finale per il secondo file (opzionale)')

args = parser.parse_args()

# Lettura dei file
df1 = pd.read_csv(args.filename1, delimiter='\t', header=None)
df2 = pd.read_csv(args.filename2, delimiter='\t', header=None)

# Creazione di array numpy per il primo file
array1 = df1.values
passi_motore1 = array1[:, 0]
intensity1 = array1[:, 1]

# Creazione di array numpy per il secondo file
array2 = df2.values
passi_motore2 = array2[:, 0]
intensity2 = array2[:, 1]

# Applica il filtro sull'intervallo per il primo file, se specificato
if args.start1 is not None or args.end1 is not None:
    mask1 = np.ones(len(passi_motore1), dtype=bool)
    if args.start1 is not None:
        mask1 &= passi_motore1 >= args.start1
    if args.end1 is not None:
        mask1 &= passi_motore1 <= args.end1
    passi_motore1 = passi_motore1[mask1]
    intensity1 = intensity1[mask1]

# Applica il filtro sull'intervallo per il secondo file, se specificato
if args.start2 is not None or args.end2 is not None:
    mask2 = np.ones(len(passi_motore2), dtype=bool)
    if args.start2 is not None:
        mask2 &= passi_motore2 >= args.start2
    if args.end2 is not None:
        mask2 &= passi_motore2 <= args.end2
    passi_motore2 = passi_motore2[mask2]
    intensity2 = intensity2[mask2]

# Esempio di plot per i due file
plt.figure(figsize=(10, 6))
plt.plot(passi_motore1, intensity1, label='Dati presi in avanti', marker='o')
plt.plot(passi_motore2, intensity2, label='Dati presi indietro', marker='x')
plt.axhline(y=2281, color='b', linestyle='-', label='Rumore')
plt.xlabel("Passi Motore")
plt.ylabel("Intensity")
plt.title("Confronto presa dati in avanti e indietro")
plt.legend()
plt.grid()
plt.show()

# Assicuriamoci che i due array abbiano la stessa dimensione per il confronto
if len(passi_motore1) != len(passi_motore2):
    raise ValueError("I due file non hanno lo stesso numero di punti dopo il filtro.")

# Calcolo della differenza tra le intensità
scarti = intensity1 - intensity2

# Esempio di plot per i due file
plt.figure(figsize=(10, 6))
for i in range(len(passi_motore1)):
    plt.plot([passi_motore1[i], passi_motore1[i] - scarti[i]], [intensity1[i], intensity1[i]], 'b-', label='Scarto' if i == 0 else "")

plt.scatter(passi_motore1, intensity1, color='blue', label='Dati 1')
plt.scatter(passi_motore2, intensity2, color='orange', label='Dati 2')
plt.axhline(y=0, color='r', linestyle='--', label='Linea zero')
plt.xlabel("Passi Motore")
plt.ylabel("Intensità")
plt.title("Scarti orizzontali tra i due file")
plt.legend()
plt.grid()
plt.show()
