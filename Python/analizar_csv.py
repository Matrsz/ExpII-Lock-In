import numpy as np
from uncertainties import ufloat
import csv
import matplotlib.pyplot as plt

filename = 'sim_out.csv'
data = np.genfromtxt(filename, delimiter=' ')

t, v, R, P = data[:,0], data[:, 1], data[:, 2], data[:, 3]

R_m = ufloat(np.mean(R), np.std(R))
P_m = ufloat(np.mean(P), np.std(P))

print("R = ", R_m)
print("P = ", P_m)

fig, axs = plt.subplots(2,1)
axs[0].plot(t, R, "b")
axs[0].set_title("Amplitud")
axs[0].set_ylim([0, np.max(R)*1.2])
axs[1].plot(t, P, "r")
axs[1].set_ylim([-1, 1])
axs[1].set_title("Fase")
plt.show()

def fourierizar(s):
    S = np.fft.fft(s)
    S = np.abs(S)
    return S

fig, axs = plt.subplots(2,2)
Ts = (t[-1]-t[0])/len(t)
f = np.fft.fftfreq(len(t), Ts)
lims = [np.min(v)*1.1, np.max(v)*1.1]
axs[0,0].plot(t, v, "b")
axs[0,0].set_title("Entrada Lockin")
axs[0,0].set_ylim(lims)
axs[0,1].plot(t, R, "r")
axs[0,1].set_ylim(lims)
axs[0,1].set_title("Salida Lockin")
axs[1,0].plot(f, fourierizar(v), "b")
axs[1,0].set_title("Entrada Lockin")
axs[1,0].set_ylim([0, np.max(fourierizar(v))*1.1])
axs[1,1].plot(f, fourierizar(R), "r")
axs[1,1].set_ylim([0, np.max(fourierizar(R))*1.1])
axs[1,1].set_title("Salida Lockin")
plt.show()


#H = R_m*np.cos(P_m) + 1j*R_m*np.sin(P_m)
H = np.mean(R)*np.cos(np.mean(P)*np.pi) + 1j*np.mean(R)*np.sin(np.mean(P)*np.pi)
Z = (H*10000)/(1-H)
print("Z = ", Z)
print("Resistencia:",np.real(Z),"\nCapacitancia:",-1/(2*np.pi*23.405138*np.imag(Z)))
print("H:",H)