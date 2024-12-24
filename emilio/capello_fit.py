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

ordini = np.array([-5, -4, -3, -2, 2, 3, 4, 5])
sintheta = np.array([-0.03276018729954389, -0.026517241354273097, -0.02010139603826872, -0.013169069330662028, 0.013474642210571723, 0.020559701288946672, 0.02638358813244988, 0.033046532169660706])
s_sintheta = np.array([0.000349996071042141, 0.0002895844836135268, 0.00022857773629836786, 0.0001653125801384797, 0.00014480256934287634, 0.0002077581618880818, 0.0002625154698215019, 0.0003266072188930084])


m, q, s_m, s_q = fit_lineare_doro(ordini, sintheta, s_sintheta)

print(f"m_sper: {m} +- {s_m}")
print(f"q_sper: {q} +- {s_q}")


lambda_ = 670*10**-9
s_lambda = 5*10**-9
larghezza_fenditura = lambda_ / m
s_larghezza_fenditura = np.sqrt((s_lambda / m)**2 + (lambda_ * s_m / m**2)**2)

print(f"Larghezza fenditura: {larghezza_fenditura * 10**6} +- {s_larghezza_fenditura*10**6} micrometri")


plt.errorbar(ordini, sintheta, c='b', yerr=s_sintheta, fmt='None')
plt.scatter(ordini, sintheta, c='r')
plt.plot(ordini, m * ordini + q, c='b', label='Fit lineare')
plt.legend()
plt.xlabel("Ordini")
plt.ylabel("sin(theta - theta0) mrad")
plt.show()