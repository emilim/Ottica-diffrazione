import matplotlib.pyplot as plt
import numpy as np

ordini = np.array([-4, -3, -2, 2, 3, 4])
theta_sperimentale_mrad = np.array([-30.941999999999997, -22.156, -14.2295, 15.470999999999998, 23.110999999999997, 31.897])
theta_fit_mrad = 

theta_0 = 0.6453607865452808
s_theta_0 = 0.05551335460498983
s_theta_sperimentale_mrad = 0
plt.scatter(ordini, theta_sperimentale_mrad)
plt.xlabel("Ordini")
plt.ylabel("sin(theta - theta0) mrad")
plt.show()