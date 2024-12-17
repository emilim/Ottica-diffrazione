import numpy as np

minimo = 33.78852285577387 #passi motore
minimo_mrad =minimo*0.0191
s_conv= 0.0191*np.sqrt((0.005/0.5)**2 + (0.5/(400*65.5))**2)
s_min_rad=np.sqrt(minimo**2*s_conv**2 + np.pow(0.0191, 2)*100/12)

print(f"minimo sperimentale: {minimo_mrad} +- {s_min_rad} mrad")