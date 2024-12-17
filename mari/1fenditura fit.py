import matplotlib.pyplot as plt
import numpy as np

ordini = np.array([-4, -3, -2, 2, 3, 4])
theta_sperimentale_mrad_0 = np.array([-30.941999999999997, -22.156, -14.2295, 15.470999999999998, 23.110999999999997, 31.897])
s_theta_sperimentale_mrad_0 = np.array([0.3142947158624779, 0.22831797063268983, 0.15260414182006307, 0.1642417565808675, 0.2375965276642299, 0.32370096505114804])
theta_fit_mrad_0 = np.array([-31.019382178683067,-22.283490876860338, -14.233058636590181, 15.526788415980471, 23.156284422689186, 31.976211890797252])
s_theta_fit_mrad_0 = np.array([0.3101944224950267, 0.22283539138873246, 0.1526373248186308, 0.1552683610212166, 0.2315633080730888, 0.3197627338839551])

theta_0 = 0.6453607865452808
s_theta_0 = 0.05551335460498983

theta_sperimentale_mrad = theta_sperimentale_mrad_0 - theta_0
s_theta_sperimentale_mrad = np.sqrt(np.pow(s_theta_sperimentale_mrad_0, 2) + np.pow(s_theta_0, 2))
theta_fit_mrad = theta_fit_mrad_0 - theta_0
s_theta_fit_mrad = np.sqrt(np.pow(s_theta_fit_mrad_0, 2) + np.pow(s_theta_0, 2))

y_sper = np.sin(theta_sperimentale_mrad*10**-3)
y_fit = np.sin(theta_fit_mrad*10**-3)

s_y_sper = np.abs(np.cos(theta_sperimentale_mrad*10**-3) * s_theta_sperimentale_mrad*10**-3)
s_y_fit = np.abs(np.cos(theta_fit_mrad*10**-3) * s_theta_fit_mrad*10**-3)

plt.errorbar(ordini, y_sper, c='b', yerr=s_y_sper, fmt='None')
plt.errorbar(ordini, y_fit, c='r', yerr=s_y_fit, fmt='None')
plt.scatter(ordini, y_sper, c='b')
plt.scatter(ordini, y_fit, c='r')

plt.xlabel("Ordini")
plt.ylabel("sin(theta - theta0) mrad")
plt.show()