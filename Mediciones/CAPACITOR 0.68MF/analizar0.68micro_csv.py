import numpy as np
from uncertainties import ufloat
import csv
import matplotlib.pyplot as plt
from scipy.signal import argrelextrema


filename = 'sim_out_pocoruido.csv'
data = np.genfromtxt(filename, delimiter=' ')

t1, v1, R1, P1 = data[:,0], data[:, 1], data[:, 2], data[:, 3]

R_m = ufloat(np.mean(R1), np.std(R1))
P_m = ufloat(np.mean(P1), np.std(P1))

H = np.mean(R1)*np.cos(np.mean(P1)*np.pi) + 1j*np.mean(R1)*np.sin(np.mean(P1)*np.pi)
Z = (H*10000)/(1-H)
C = []
C.append(1/(2*np.pi*23.405138*np.imag(Z)))

################################################

filename = 'sim_out_4V4000.csv'
data = np.genfromtxt(filename, delimiter=' ')

t1, v1, R1, P1 = data[:,0], data[:, 1], data[:, 2], data[:, 3]

R_m = ufloat(np.mean(R1), np.std(R1))
P_m = ufloat(np.mean(P1), np.std(P1))

H = np.mean(R1)*np.cos(np.mean(P1)*np.pi) + 1j*np.mean(R1)*np.sin(np.mean(P1)*np.pi)
Z = (H*10000)/(1-H)
C.append(1/(2*np.pi*23.405138*np.imag(Z)))

################################################

filename = 'sim_out_1V.csv'
data = np.genfromtxt(filename, delimiter=' ')

t1, v1, R1, P1 = data[:,0], data[:, 1], data[:, 2], data[:, 3]

R_m = ufloat(np.mean(R1), np.std(R1))
P_m = ufloat(np.mean(P1), np.std(P1))

H = np.mean(R1)*np.cos(np.mean(P1)*np.pi) + 1j*np.mean(R1)*np.sin(np.mean(P1)*np.pi)
Z = (H*10000)/(1-H)
C.append(1/(2*np.pi*23.405138*np.imag(Z)))

################################################

filename = 'sim_out_0.8V.csv'
data = np.genfromtxt(filename, delimiter=' ')

t1, v1, R1, P1 = data[:,0], data[:, 1], data[:, 2], data[:, 3]

R_m = ufloat(np.mean(R1), np.std(R1))
P_m = ufloat(np.mean(P1), np.std(P1))

H = np.mean(R1)*np.cos(np.mean(P1)*np.pi) + 1j*np.mean(R1)*np.sin(np.mean(P1)*np.pi)
Z = (H*10000)/(1-H)
C.append(1/(2*np.pi*23.405138*np.imag(Z)))

################################################

filename = 'sim_out_0.6V.csv'
data = np.genfromtxt(filename, delimiter=' ')

t1, v1, R1, P1 = data[:,0], data[:, 1], data[:, 2], data[:, 3]

R_m = ufloat(np.mean(R1), np.std(R1))
P_m = ufloat(np.mean(P1), np.std(P1))

H = np.mean(R1)*np.cos(np.mean(P1)*np.pi) + 1j*np.mean(R1)*np.sin(np.mean(P1)*np.pi)
Z = (H*10000)/(1-H)
C.append(1/(2*np.pi*23.405138*np.imag(Z)))

################################################

filename = 'sim_out_0.4v.csv'
data = np.genfromtxt(filename, delimiter=' ')

t1, v1, R1, P1 = data[:,0], data[:, 1], data[:, 2], data[:, 3]

R_m = ufloat(np.mean(R1), np.std(R1))
P_m = ufloat(np.mean(P1), np.std(P1))

H = np.mean(R1)*np.cos(np.mean(P1)*np.pi) + 1j*np.mean(R1)*np.sin(np.mean(P1)*np.pi)
Z = (H*10000)/(1-H)
C.append(1/(2*np.pi*23.405138*np.imag(Z)))

################################################

SNR = []

SNR.append(0)
SNR.append(1)
SNR.append(1/4)
SNR.append(0.8/4)
SNR.append(0.6/4)
SNR.append(0.4/4)
#No puse el valor de 0.2/4 porque me da un valor negativo de capacidad(encima 9e-6)
print("C = ",C)
print("SNR = ",SNR)
plt.plot(SNR,C,'r',marker="o")
plt.axhline(y=6.8e-7, xmin=0, xmax=1)
plt.xlabel("SNR")
plt.ylabel("CAPACIDAD")
plt.grid()
plt.show()
