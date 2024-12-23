import matplotlib.pyplot as plt
import numpy as np

def fit_lineare_doro(x, y, sigma_y):
    W = 1 / sigma_y**2  # Pesi
    Delta = np.sum(W) * np.sum(W * x**2) - (np.sum(W * x))**2
    m = (np.sum(W) * np.sum(W * x * y) - np.sum(W * x) * np.sum(W * y)) / Delta
    q = (np.sum(W * x**2) * np.sum(W * y) - np.sum(W * x) * np.sum(W * x * y)) / Delta
    sigma_m = np.sqrt(np.sum(W) / Delta)
    sigma_q = np.sqrt(np.sum(W * x**2) / Delta)
    return m, q, sigma_m, sigma_q

ordini = np.array([-4, -3, -2, 2, 3, 4])
theta_sperimentale_mrad_0 = np.array([-30.941999999999997, -22.156, -14.2295, 15.470999999999998, 23.110999999999997, 31.897])
s_theta_sperimentale_mrad_0 = np.array([0.39315459874493414, 0.28413693242741894, 0.18731449247901996, 0.20229359141745576, 0.2959322185309574, 0.4050538458385091])
theta_fit_mrad_0 = np.array([-31.019382178683067,-22.283490876860338, -14.233058636590181, 15.526788415980471, 23.156284422689186, 31.976211890797252])
s_theta_fit_mrad_0 = np.array([0.390242672742653, 0.22283539138873246, 0.17906083218398203, 0.19533657116565725, 0.29132013822318353, 0.4022801684278837])

theta_0 = 0.606361032418853 
s_theta_0 = 0.00871573335376667

theta_sperimentale_mrad = theta_sperimentale_mrad_0 - theta_0
s_theta_sperimentale_mrad = np.sqrt(np.power(s_theta_sperimentale_mrad_0, 2) + np.power(s_theta_0, 2))
theta_fit_mrad = theta_fit_mrad_0 - theta_0
s_theta_fit_mrad = np.sqrt(np.power(s_theta_fit_mrad_0, 2) + np.power(s_theta_0, 2))

y_sper = np.sin(theta_sperimentale_mrad*10**-3)
y_fit = np.sin(theta_fit_mrad*10**-3)
s_y_sper = np.abs(np.cos(theta_sperimentale_mrad*10**-3) * s_theta_sperimentale_mrad*10**-3)
print(f"ysper:{s_y_sper} ")
s_y_fit = np.abs(np.cos(theta_fit_mrad*10**-3) * s_theta_fit_mrad*10**-3)
print(f"ysper:{s_y_fit} ")
m_sper, q_sper, s_m_sper, s_q_sper = fit_lineare_doro(ordini, y_sper, s_y_sper)
m_fit, q_doro_fit, s_m_fit, s_q_fit = fit_lineare_doro(ordini, y_fit, s_y_fit)

print(f"m_sper: {m_sper} +- {s_m_sper}")
print(f"q_sper: {q_sper} +- {s_q_sper}")
print(f"m_fit: {m_fit} +- {s_m_fit}")
print(f"q_fit: {q_doro_fit} +- {s_q_fit}")

chi_quadro_sper = np.sum(((y_sper - m_sper * ordini - q_sper) / s_y_sper)**2)
chi_quadro_fit = np.sum(((y_fit - m_fit * ordini - q_doro_fit) / s_y_fit)**2)

print(f"Chi quadro sper: {chi_quadro_sper}")
print(f"Chi quadro fit: {chi_quadro_fit}")

lambda_ = 670*10**-9
s_lambda = 5*10**-9
larghezza_fenditura_sper = lambda_ / m_sper
s_larghezza_fenditura_sper = np.sqrt((s_lambda / m_sper)**2 + (lambda_ * s_m_sper / m_sper**2)**2)
larghezza_fenditura_fit = lambda_ / m_fit
s_larghezza_fenditura_fit = np.sqrt((s_lambda / m_fit)**2 + (lambda_ * s_m_fit / m_fit**2)**2)

print(f"Larghezza fenditura sper: {larghezza_fenditura_sper} +- {s_larghezza_fenditura_sper}")
print(f"Larghezza fenditura fit: {larghezza_fenditura_fit} +- {s_larghezza_fenditura_fit}")

compatibility = np.abs(m_sper - m_fit) / np.sqrt(s_m_sper**2 + s_m_fit**2)
print(f"Compatibilità: {compatibility}")

plt.errorbar(ordini, y_sper, c='b', yerr=s_y_sper, fmt='None')
plt.errorbar(ordini, y_fit, c='r', yerr=s_y_fit, fmt='None')
plt.scatter(ordini, y_sper, c='b')
plt.scatter(ordini, y_fit, c='r')
plt.plot(ordini, m_sper * ordini + q_sper, color='red', label='Sperimentale ')
plt.plot(ordini, m_fit * ordini + q_doro_fit,  color='blue', label='Fit parabolico' )
plt.xlabel("Ordini")
plt.ylabel("sin(theta - theta0) mrad")
plt.show()