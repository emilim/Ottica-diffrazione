import numpy as np

massimo = -667 #passi motore
massimo_mrad =massimo*0.0191
s_conv= 0.0191*np.sqrt((0.005/0.5)**2 + (0.5/(65.5))**2)
s_min_rad=np.sqrt(massimo**2*s_conv**2 + np.pow(0.0191, 2)*9/12) #s_x=deltapassi/sqrt(12)

print(f"massimo sperimentale: {massimo_mrad} +- {s_min_rad} mrad")