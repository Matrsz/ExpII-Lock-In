import numpy as np
from uncertainties import ufloat
import csv
import matplotlib.pyplot as plt
from scipy.signal import argrelextrema
R = 470
f = 30 #no la uso igual

Resistencia = []

filename = 'sim_out_4V4000.csv'
data = np.genfromtxt(filename, delimiter=' ')

t1, v1, R1, P1 = data[:,0], data[:, 1], data[:, 2], data[:, 3]

R_m = ufloat(np.mean(R1), np.std(R1))
P_m = ufloat(np.mean(P1), np.std(P1))
H = np.mean(R1)*np.cos(np.mean(P1)*np.pi) + 1j*np.mean(R1)*np.sin(np.mean(P1)*np.pi)
Z = (H*R)/(1-H)
print(H)
print(Z)
Resistencia.append(np.abs(Z))

################################################

filename = 'sim_out_1V4000.csv'
data = np.genfromtxt(filename, delimiter=' ')

t1, v1, R1, P1 = data[:,0], data[:, 1], data[:, 2], data[:, 3]

R_m = ufloat(np.mean(R1), np.std(R1))
P_m = ufloat(np.mean(P1), np.std(P1))

H = np.mean(R1)*np.cos(np.mean(P1)*np.pi) + 1j*np.mean(R1)*np.sin(np.mean(P1)*np.pi)
Z = (H*R)/(1-H)
Resistencia.append(np.abs(Z))

################################################

filename = 'sim_out_0.8V4000.csv'
data = np.genfromtxt(filename, delimiter=' ')

t1, v1, R1, P1 = data[:,0], data[:, 1], data[:, 2], data[:, 3]

R_m = ufloat(np.mean(R1), np.std(R1))
P_m = ufloat(np.mean(P1), np.std(P1))

H = np.mean(R1)*np.cos(np.mean(P1)*np.pi) + 1j*np.mean(R1)*np.sin(np.mean(P1)*np.pi)
Z = (H*R)/(1-H)
Resistencia.append(np.abs(Z))

################################################

filename = 'sim_out_0.6V4000.csv'
data = np.genfromtxt(filename, delimiter=' ')

t1, v1, R1, P1 = data[:,0], data[:, 1], data[:, 2], data[:, 3]

R_m = ufloat(np.mean(R1), np.std(R1))
P_m = ufloat(np.mean(P1), np.std(P1))

H = np.mean(R1)*np.cos(np.mean(P1)*np.pi) + 1j*np.mean(R1)*np.sin(np.mean(P1)*np.pi)
Z = (H*R)/(1-H)
Resistencia.append(np.abs(Z))

################################################

filename = 'sim_out_0.4V4000.csv'
data = np.genfromtxt(filename, delimiter=' ')

t1, v1, R1, P1 = data[:,0], data[:, 1], data[:, 2], data[:, 3]

R_m = ufloat(np.mean(R1), np.std(R1))
P_m = ufloat(np.mean(P1), np.std(P1))

H = np.mean(R1)*np.cos(np.mean(P1)*np.pi) + 1j*np.mean(R1)*np.sin(np.mean(P1)*np.pi)
Z = (H*R)/(1-H)
Resistencia.append(np.abs(Z))

################################################

filename = 'sim_out_0.2V4000.csv'
data = np.genfromtxt(filename, delimiter=' ')

t1, v1, R1, P1 = data[:,0], data[:, 1], data[:, 2], data[:, 3]

R_m = ufloat(np.mean(R1), np.std(R1))
P_m = ufloat(np.mean(P1), np.std(P1))

H = np.mean(R1)*np.cos(np.mean(P1)*np.pi) + 1j*np.mean(R1)*np.sin(np.mean(P1)*np.pi)
Z = (H*R)/(1-H)
Resistencia.append(np.abs(Z))

################################################

Resistencia2 = []

filename = 'sim_out_4V2000.csv'
data = np.genfromtxt(filename, delimiter=' ')

t1, v1, R1, P1 = data[:,0], data[:, 1], data[:, 2], data[:, 3]

R_m = ufloat(np.mean(R1), np.std(R1))
P_m = ufloat(np.mean(P1), np.std(P1))
H = np.mean(R1)*np.cos(np.mean(P1)*np.pi) + 1j*np.mean(R1)*np.sin(np.mean(P1)*np.pi)
Z = (H*R)/(1-H)
print(H)
print(Z)
Resistencia2.append(np.abs(Z))

################################################

filename = 'sim_out_1V2000.csv'
data = np.genfromtxt(filename, delimiter=' ')

t1, v1, R1, P1 = data[:,0], data[:, 1], data[:, 2], data[:, 3]

R_m = ufloat(np.mean(R1), np.std(R1))
P_m = ufloat(np.mean(P1), np.std(P1))

H = np.mean(R1)*np.cos(np.mean(P1)*np.pi) + 1j*np.mean(R1)*np.sin(np.mean(P1)*np.pi)
Z = (H*R)/(1-H)
Resistencia2.append(np.abs(Z))

################################################

filename = 'sim_out_0.8V2000.csv'
data = np.genfromtxt(filename, delimiter=' ')

t1, v1, R1, P1 = data[:,0], data[:, 1], data[:, 2], data[:, 3]

R_m = ufloat(np.mean(R1), np.std(R1))
P_m = ufloat(np.mean(P1), np.std(P1))

H = np.mean(R1)*np.cos(np.mean(P1)*np.pi) + 1j*np.mean(R1)*np.sin(np.mean(P1)*np.pi)
Z = (H*R)/(1-H)
Resistencia2.append(np.abs(Z))

################################################

filename = 'sim_out_0.6V2000.csv'
data = np.genfromtxt(filename, delimiter=' ')

t1, v1, R1, P1 = data[:,0], data[:, 1], data[:, 2], data[:, 3]

R_m = ufloat(np.mean(R1), np.std(R1))
P_m = ufloat(np.mean(P1), np.std(P1))

H = np.mean(R1)*np.cos(np.mean(P1)*np.pi) + 1j*np.mean(R1)*np.sin(np.mean(P1)*np.pi)
Z = (H*R)/(1-H)
Resistencia2.append(np.abs(Z))

################################################

filename = 'sim_out_0.4V2000.csv'
data = np.genfromtxt(filename, delimiter=' ')

t1, v1, R1, P1 = data[:,0], data[:, 1], data[:, 2], data[:, 3]

R_m = ufloat(np.mean(R1), np.std(R1))
P_m = ufloat(np.mean(P1), np.std(P1))

H = np.mean(R1)*np.cos(np.mean(P1)*np.pi) + 1j*np.mean(R1)*np.sin(np.mean(P1)*np.pi)
Z = (H*R)/(1-H)
Resistencia2.append(np.abs(Z))

################################################

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

filename = 'sim_out_4V1000.csv'
data = np.genfromtxt(filename, delimiter=' ')

t1, v1, R1, P1 = data[:,0], data[:, 1], data[:, 2], data[:, 3]

R_m = ufloat(np.mean(R1), np.std(R1))
P_m = ufloat(np.mean(P1), np.std(P1))
H = np.mean(R1)*np.cos(np.mean(P1)*np.pi) + 1j*np.mean(R1)*np.sin(np.mean(P1)*np.pi)
Z = (H*R)/(1-H)
print(H)
print(Z)
Resistencia3.append(np.abs(Z))

################################################

filename = 'sim_out_1V1000.csv'
data = np.genfromtxt(filename, delimiter=' ')

t1, v1, R1, P1 = data[:,0], data[:, 1], data[:, 2], data[:, 3]

R_m = ufloat(np.mean(R1), np.std(R1))
P_m = ufloat(np.mean(P1), np.std(P1))

H = np.mean(R1)*np.cos(np.mean(P1)*np.pi) + 1j*np.mean(R1)*np.sin(np.mean(P1)*np.pi)
Z = (H*R)/(1-H)
Resistencia3.append(np.abs(Z))

################################################

filename = 'sim_out_0.8V1000.csv'
data = np.genfromtxt(filename, delimiter=' ')

t1, v1, R1, P1 = data[:,0], data[:, 1], data[:, 2], data[:, 3]

R_m = ufloat(np.mean(R1), np.std(R1))
P_m = ufloat(np.mean(P1), np.std(P1))

H = np.mean(R1)*np.cos(np.mean(P1)*np.pi) + 1j*np.mean(R1)*np.sin(np.mean(P1)*np.pi)
Z = (H*R)/(1-H)
Resistencia3.append(np.abs(Z))

################################################

filename = 'sim_out_0.6V1000.csv'
data = np.genfromtxt(filename, delimiter=' ')

t1, v1, R1, P1 = data[:,0], data[:, 1], data[:, 2], data[:, 3]

R_m = ufloat(np.mean(R1), np.std(R1))
P_m = ufloat(np.mean(P1), np.std(P1))

H = np.mean(R1)*np.cos(np.mean(P1)*np.pi) + 1j*np.mean(R1)*np.sin(np.mean(P1)*np.pi)
Z = (H*R)/(1-H)
Resistencia3.append(np.abs(Z))

################################################

filename = 'sim_out_0.4V1000.csv'
data = np.genfromtxt(filename, delimiter=' ')

t1, v1, R1, P1 = data[:,0], data[:, 1], data[:, 2], data[:, 3]

R_m = ufloat(np.mean(R1), np.std(R1))
P_m = ufloat(np.mean(P1), np.std(P1))

H = np.mean(R1)*np.cos(np.mean(P1)*np.pi) + 1j*np.mean(R1)*np.sin(np.mean(P1)*np.pi)
Z = (H*R)/(1-H)
Resistencia3.append(np.abs(Z))

################################################

filename = 'sim_out_0.2V1000.csv'
data = np.genfromtxt(filename, delimiter=' ')

t1, v1, R1, P1 = data[:,0], data[:, 1], data[:, 2], data[:, 3]

R_m = ufloat(np.mean(R1), np.std(R1))
P_m = ufloat(np.mean(P1), np.std(P1))

H = np.mean(R1)*np.cos(np.mean(P1)*np.pi) + 1j*np.mean(R1)*np.sin(np.mean(P1)*np.pi)
Z = (H*R)/(1-H)
Resistencia3.append(np.abs(Z))

################################################

SNR = []

SNR.append(1)
SNR.append(1/4)
SNR.append(0.8/4)
SNR.append(0.6/4)
SNR.append(0.4/4)
SNR.append(0.2/4)

#SNR = [14.2, 3.2, 1.3, -1.5, -4.4, -15.5]
SNR = [13.399485222497487, 2.9242719945388673, 0.6488520868543456, 
-5.357905252717508, -8.113899133972234, -11.156481168356017]


plt.plot(SNR,Resistencia,'b',marker="o")
plt.plot(SNR,Resistencia2,'g',marker="v")
plt.plot(SNR,Resistencia3,'y',marker="s")
plt.axhline(y=470-23.5, xmin=0, xmax=1,color = 'r',linestyle = '--')
plt.axhline(y=470+23.5, xmin=0, xmax=1,color = 'r',linestyle = '--')
plt.legend(['N = 4000', 'N = 2000', 'N = 1000'])
plt.xlabel("SNR[dB]")
plt.ylabel("RESISTENCIA[Ω]")
plt.grid()
plt.show()
