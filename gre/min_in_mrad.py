import pandas as pd
import numpy as np

# Funzione per calcolare minimo_mrad, incertezza e ordine di grandezza
def calcola_valori(minimo):
    minimo_mrad = minimo * 0.0191
    s_conv = (1/400)*np.sqrt((0.005/65.5)**2 + (((-(0.5/(65.5**2)))**2)*(0.5**2)))
    s_min_rad = np.sqrt(minimo**2 * s_conv**2 + np.power(0.0191, 2)*100/12)  # s_x = deltapassi/sqrt(12)
    
    return minimo_mrad, s_min_rad

# Caricare il file di testo con i dati
file_path = 'minimi_3fenditure.txt'  # Cambia il percorso del file di input
data = pd.read_csv(file_path, sep='\t', header=0, names=['Passi', 'Intensità'])

# Inizializzare una lista per i risultati
risultati = []
index = 1

# Calcolare i valori per ogni riga del DataFrame
for index, row in data.iterrows():
    passi = row['Passi']
    minimo_mrad, s_min_rad = calcola_valori(passi)
      
    if passi < 0:  # Controlla se il valore è sotto zero
        ordine_grandezza = -((7/3)-(index/3))
    
    if passi > 0:
        ordine_grandezza = (index/3)-2
        
    
 
    # Aggiungere i risultati alla lista
    risultati.append([ordine_grandezza, minimo_mrad, s_min_rad])

# Creare un DataFrame con i risultati
risultati_df = pd.DataFrame(risultati, columns=['Ordine_Grandezza', 'Minimo_mrad', 'Incertezza_mrad'])

# Salvare i risultati in un nuovo file
output_file_path = 'risultati_minimi.txt'  # Cambia il percorso del file di output
risultati_df.to_csv(output_file_path, index=False, sep='\t')

print(f"Risultati salvati in {output_file_path}")
