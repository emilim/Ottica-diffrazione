import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

file_path = "trefenditure_avanti.xls"
data = pd.read_csv(file_path, header=None, delimiter='\t')


posizione = data[0]  
intensità = data[1]  

min_posizioni = []
min_intensità = []

for i in range(1, len(intensità) - 1):
    
    if intensità[i] < intensità[i-1] and intensità[i] < intensità[i+1]:
        min_posizioni.append(posizione[i])
        min_intensità.append(intensità[i])


plt.figure(figsize=(10, 6))
plt.plot(posizione, intensità, label="Figura di Interferenza")
plt.scatter(min_posizioni, min_intensità, color='red', label='Minimi sperimentali', zorder=5)
plt.xlabel("Posizione")
plt.ylabel("Intensità")
plt.title("Figura di Interferenza: Minimi Sperimentali")
plt.legend()
plt.grid()
plt.show()


output_file_path = "minimi_3fenditure.txt" 
with open(output_file_path, 'w') as file:
    
    file.write("Posizione \t Intensita'\n")
    

    for pos, inten in zip(min_posizioni, min_intensità):
        file.write(f"{pos} \t {inten}\n")

