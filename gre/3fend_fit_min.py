import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

def fit_lineare_doro(x, y, sigma_y):
    W = 1 / sigma_y**2  # Pesi
    Delta = np.sum(W) * np.sum(W * x**2) - (np.sum(W * x))**2
    m = (np.sum(W) * np.sum(W * x * y) - np.sum(W * x) * np.sum(W * y)) / Delta
    q = (np.sum(W * x**2) * np.sum(W * y) - np.sum(W * x) * np.sum(W * x * y)) / Delta
    sigma_m = np.sqrt(np.sum(W) / Delta)
    sigma_q = np.sqrt(np.sum(W * x**2) / Delta)
    return m, q, sigma_m, sigma_q


file_path = 'risultati_minimi.txt' 
data = pd.read_csv(file_path, sep='\t', header=None, names=['Ordine', 'Angolo', 'Incertezza'])

ordini = data['Ordine'].values
#theta_sperimentale_mrad_0 = data['Angolo'].values
#s_theta_sperimentale_mrad_0 = data['Incertezza'].values
def parse_angle_and_uncertainty(data):
    try:
        parts = data.split('\t')  # Divide la stringa nei valori separati da uno spazio
        angle = float(parts[0])  # Il primo valore è l'angolo
        uncertainty = float(parts[1])  # Il secondo valore è l'incertezza
        return angle, uncertainty
    except ValueError:
        # Se non possiamo convertire in float, restituiamo None per indicare errore
        return None, None

# Usa la funzione per estrarre angoli e incertezze
theta_sperimentale_mrad_0, s_theta_sperimentale_mrad_0 = zip(*data['Angolo'].map(parse_angle_and_uncertainty))

# Converte in array numpy
theta_sperimentale_mrad_0 = np.array(theta_sperimentale_mrad_0)
s_theta_sperimentale_mrad_0 = np.array(s_theta_sperimentale_mrad_0)

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