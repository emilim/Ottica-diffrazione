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

#ordini = np.array([-6, -5, -4, -3, -2, 2, 3, 4, 5, 6])
#theta_sperimentale_mrad_0 = np.array([-17.6102, -13.083499999999999, -9.3017, -5.6918,-2.9987, 4.7368, 8.289399999999999, 12.128499999999999, 15.5092, 18.5461])
#s_theta_sperimentale_mrad_0 = np.array([0.2221636333816838, 0.16542741535116073, 0.11818429927946887, 0.07349196705419116, 0.04119245332492908, 0.061844904363421584, 0.10558934903978164, 0.15347784897833366, 0.19581503233308176, 0.23390678554204905])

ordini = np.array([-4, -3,-2, 2, 3, 4])
theta_sperimentale_mrad_0 = np.array([ -9.3017, -5.6918,-2.9987, 4.7368, 8.289399999999999, 12.128499999999999])
s_theta_sperimentale_mrad_0 = np.array([ 0.11818429927946887, 0.07349196705419116, 0.04119245332492908, 0.061844904363421584, 0.10558934903978164, 0.15347784897833366])

theta_0 = 0.8976999999999999 
s_theta_0 =0.020028807732656324

theta_mrad = theta_sperimentale_mrad_0 - theta_0
s_theta_mrad = np.sqrt(np.power(s_theta_sperimentale_mrad_0, 2) + np.power(s_theta_0, 2))

y_sper = np.sin(theta_mrad*10**-3)

s_y_sper = np.abs(np.cos(theta_mrad*10**-3) * s_theta_mrad*10**-3)
print(f"y_sper +- s_y_sper={y_sper} +- {s_y_sper}")
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
plt.plot(ordini, m_sper * ordini + q_sper, c='b', label='fit lineare')
plt.legend()
plt.xlabel("Ordini")
plt.ylabel("sin(theta - theta0) mrad")
plt.show()

a=ordini*lambda_/y_sper
s_a=a*np.sqrt((s_lambda/lambda_)**2 + (s_y_sper/y_sper)**2)
a_medio=np.mean(a)
print(f"ampiezza fend:{a} +- {s_a}\n")
