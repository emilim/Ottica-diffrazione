import numpy as np

s_conv= 0.0191*np.sqrt((0.005/0.5)**2 + (0.5/(400*65.5))**2)

theta_0_passim = -70.5 #passi motore
theta_0 = theta_0_passim*0.0191
s_theta0 = np.sqrt(theta_0_passim**2*s_conv**2 + np.power(0.0191, 2)*100/12)

passim = -1786 #passi motore
theta = passim*0.0191
s_theta = np.sqrt(passim**2*s_conv**2 + np.power(0.0191, 2)*100/12)

thetadiff = theta - theta_0
s_thetadiff = np.sqrt(np.power(s_theta, 2) + np.power(s_theta0, 2))
sintheta = np.sin(thetadiff*10**-3)
s_sintheta = np.abs(np.cos(thetadiff*10**-3) * s_thetadiff*10**-3)

print(f"minimo sperimentale: {theta} +- {s_theta} mrad")
print(f"sin(theta-theta0): {sintheta} +- {s_sintheta}")
