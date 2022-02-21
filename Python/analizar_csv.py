import numpy as np
from uncertainties import ufloat
import csv
import matplotlib.pyplot as plt

filename = 'sim_out.csv'
data = np.genfromtxt(filename, delimiter=' ')

t, R, P = data[:,0], data[:, 1], data[:, 2]

R_m = ufloat(np.mean(R), np.std(R))
P_m = ufloat(np.mean(P), np.std(P))

print("R = ", R_m)
print("P = ", P_m)

sfig, axs = plt.subplots(2,1)
axs[0].plot(t,R, "b")
axs[0].set_title("Amplitud")
axs[0].set_ylim([0, np.max(R)*1.2])
axs[1].plot(t, P, "r")
axs[1].set_ylim([-1, 1])
axs[1].set_title("Fase")
plt.show()

#H = R_m*np.cos(P_m) + 1j*R_m*np.sin(P_m)
H = np.mean(R)*np.cos(np.mean(P)*np.pi) + 1j*np.mean(R)*np.sin(np.mean(P)*np.pi)
Z = H / (1-H) * 470
print("Resistencia:",np.real(Z),"\nCapacitancia:",1/(2*np.pi*150*np.imag(Z)))
print("H:",H)