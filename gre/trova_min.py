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
    
    if intensità[i] < intensità[i-1] and intensità[i] < intensità[i+1] and -1200 <= posizione[i] <= 1225:
        min_posizioni.append(posizione[i])
        min_intensità.append(intensità[i])

i=0        
while i < len(min_posizioni) - 1:        
    if min_posizioni[i+1]-min_posizioni[i]<=100:
        media_p=(min_posizioni[i+1]+min_posizioni[i])/2
        media_i=(min_intensità[i+1]+min_intensità[i])/2
        min_posizioni[i]=media_p
        del min_posizioni[i+1]
        min_intensità[i]=media_i
        del min_intensità[i+1]
    else: i+=1    

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

