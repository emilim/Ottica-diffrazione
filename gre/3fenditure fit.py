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

ordini = np.array([-4, -3, -2, -1, 1, 2, 3, 4])
theta_sperimentale_mrad_0 = np.array([-12.7397, -9.359, -5.9783, -2.5976, 4.1065, 7.4872, 10.9252, 14.3059   ])
s_theta_sperimentale_mrad_0 = np.array([0.1284665801611518, 0.0950406623525641, 0.062029247983982745, 0.030795495343225905, 0.04427129845163428, 0.07667753452129913, 0.11049728718234857, 0.14401235890469288])

theta_0 = 0.7830999999999999
s_theta_0 = 0.018301155273445377

theta_sperimentale_mrad = theta_sperimentale_mrad_0 - theta_0
s_theta_sperimentale_mrad = np.sqrt(np.power(s_theta_sperimentale_mrad_0, 2) + np.power(s_theta_0, 2))


y_sper = np.sin(theta_sperimentale_mrad*10**-3)

s_y_sper = np.abs(np.cos(theta_sperimentale_mrad*10**-3) * s_theta_sperimentale_mrad*10**-3)

m_sper, q_sper, s_m_sper, s_q_sper = fit_lineare_doro(ordini, y_sper, s_y_sper)

print(f"m_sper: {m_sper} +- {s_m_sper}")
print(f"q_sper: {q_sper} +- {s_q_sper}")

chi_quadro_sper = np.sum(((y_sper - m_sper * ordini - q_sper) / s_y_sper)**2)

print(f"Chi quadro sper: {chi_quadro_sper}")

lambda_ = 670*10**-9
s_lambda = 5*10**-9
larghezza_fenditura_sper = lambda_ / m_sper
s_larghezza_fenditura_sper = np.sqrt((s_lambda / m_sper)**2 + (lambda_ * s_m_sper / m_sper**2)**2)

print(f"Larghezza fenditura sper: {larghezza_fenditura_sper} +- {s_larghezza_fenditura_sper}")


plt.errorbar(ordini, y_sper, c='b', yerr=s_y_sper, fmt='None')
plt.scatter(ordini, y_sper, c='b')

plt.plot(ordini, m_sper * ordini + q_sper, c='b', label='Sperimentale')

plt.xlabel("Ordini")
plt.ylabel("sin(theta - theta0) mrad")
plt.show()