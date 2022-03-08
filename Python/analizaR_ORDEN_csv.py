import numpy as np
from uncertainties import ufloat
import csv
import matplotlib.pyplot as plt
from scipy.signal import argrelextrema
R = 470
f = 30 #no la uso igual

Resistencia = []

filename = 'sim_out_0.2V1000.csv'
data = np.genfromtxt(filename, delimiter=' ')

t1, v1, R1, P1 = data[:,0], data[:, 1], data[:, 2], data[:, 3]

R_m = ufloat(np.mean(R1), np.std(R1))
P_m = ufloat(np.mean(P1), np.std(P1))

H = np.mean(R1)*np.cos(np.mean(P1)*np.pi) + 1j*np.mean(R1)*np.sin(np.mean(P1)*np.pi)
Z = (H*R)/(1-H)
Resistencia.append(np.abs(Z))

################################################

Resistencia2 = []


filename = 'sim_out_0.2V2000.csv'
data = np.genfromtxt(filename, delimiter=' ')

t1, v1, R1, P1 = data[:,0], data[:, 1], data[:, 2], data[:, 3]

R_m = ufloat(np.mean(R1), np.std(R1))
P_m = ufloat(np.mean(P1), np.std(P1))

H = np.mean(R1)*np.cos(np.mean(P1)*np.pi) + 1j*np.mean(R1)*np.sin(np.mean(P1)*np.pi)
Z = (H*R)/(1-H)
Resistencia2.append(np.abs(Z))

################################################

Resistencia3 = []

filename = 'sim_out_0.2V4000.csv'
data = np.genfromtxt(filename, delimiter=' ')

t1, v1, R1, P1 = data[:,0], data[:, 1], data[:, 2], data[:, 3]

R_m = ufloat(np.mean(R1), np.std(R1))
P_m = ufloat(np.mean(P1), np.std(P1))

H = np.mean(R1)*np.cos(np.mean(P1)*np.pi) + 1j*np.mean(R1)*np.sin(np.mean(P1)*np.pi)
Z = (H*R)/(1-H)
Resistencia3.append(np.abs(Z))

################################################

Resistencia4 = []

filename = 'sim_out_0.2V8000.csv'
data = np.genfromtxt(filename, delimiter=' ')

t1, v1, R1, P1 = data[:,0], data[:, 1], data[:, 2], data[:, 3]

R_m = ufloat(np.mean(R1), np.std(R1))
P_m = ufloat(np.mean(P1), np.std(P1))

H = np.mean(R1)*np.cos(np.mean(P1)*np.pi) + 1j*np.mean(R1)*np.sin(np.mean(P1)*np.pi)
Z = (H*R)/(1-H)
Resistencia4.append(np.abs(Z))

################################################

Resistencia5 = []

filename = 'sim_out_0.2V16000.csv'
data = np.genfromtxt(filename, delimiter=' ')

t1, v1, R1, P1 = data[:,0], data[:, 1], data[:, 2], data[:, 3]

R_m = ufloat(np.mean(R1), np.std(R1))
P_m = ufloat(np.mean(P1), np.std(P1))

H = np.mean(R1)*np.cos(np.mean(P1)*np.pi) + 1j*np.mean(R1)*np.sin(np.mean(P1)*np.pi)
Z = (H*R)/(1-H)
Resistencia5.append(np.abs(Z))

################################################

SNR = []

SNR.append(0.2/4)

N =[1000,2000,4000,8000,16000]
Resistencia = [Resistencia, Resistencia2, Resistencia3, Resistencia4, Resistencia5]
plt.plot(N,Resistencia, marker="p")
plt.axhline(y=470-23.5, xmin=0, xmax=1,color = 'k',linestyle = ':')
plt.axhline(y=470, xmin=0, xmax=1,color = 'k',linestyle = '--')
plt.axhline(y=470+23.5, xmin=0, xmax=1,color = 'k',linestyle = ':')
plt.xlabel("N")
plt.title("Resistencia Medida a SNR = -22.5 dB")
plt.ylabel("RESISTENCIA [Ω]")
plt.grid()
plt.tight_layout()
plt.show()
