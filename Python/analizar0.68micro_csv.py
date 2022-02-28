import numpy as np
from uncertainties import ufloat
import csv
import matplotlib.pyplot as plt
from scipy.signal import argrelextrema

C = []

filename = 'Csim_out_4V.csv'
data = np.genfromtxt(filename, delimiter=' ')

t1, v1, R1, P1 = data[:,0], data[:, 1], data[:, 2], data[:, 3]

R_m = ufloat(np.mean(R1), np.std(R1))
P_m = ufloat(np.mean(P1), np.std(P1))

H = np.mean(R1)*np.cos(np.mean(P1)*np.pi) + 1j*np.mean(R1)*np.sin(np.mean(P1)*np.pi)
Z = (H*10000)/(1-H)
C.append(1/(2*np.pi*23.405138*np.imag(Z)))

################################################

filename = 'Csim_out_1V.csv'
data = np.genfromtxt(filename, delimiter=' ')

t1, v1, R1, P1 = data[:,0], data[:, 1], data[:, 2], data[:, 3]

R_m = ufloat(np.mean(R1), np.std(R1))
P_m = ufloat(np.mean(P1), np.std(P1))

H = np.mean(R1)*np.cos(np.mean(P1)*np.pi) + 1j*np.mean(R1)*np.sin(np.mean(P1)*np.pi)
Z = (H*10000)/(1-H)
C.append(1/(2*np.pi*23.405138*np.imag(Z)))

################################################

filename = 'Csim_out_0.8V.csv'
data = np.genfromtxt(filename, delimiter=' ')

t1, v1, R1, P1 = data[:,0], data[:, 1], data[:, 2], data[:, 3]

R_m = ufloat(np.mean(R1), np.std(R1))
P_m = ufloat(np.mean(P1), np.std(P1))

H = np.mean(R1)*np.cos(np.mean(P1)*np.pi) + 1j*np.mean(R1)*np.sin(np.mean(P1)*np.pi)
Z = (H*10000)/(1-H)
C.append(1/(2*np.pi*23.405138*np.imag(Z)))

################################################

filename = 'Csim_out_0.6V.csv'
data = np.genfromtxt(filename, delimiter=' ')

t1, v1, R1, P1 = data[:,0], data[:, 1], data[:, 2], data[:, 3]

R_m = ufloat(np.mean(R1), np.std(R1))
P_m = ufloat(np.mean(P1), np.std(P1))

H = np.mean(R1)*np.cos(np.mean(P1)*np.pi) + 1j*np.mean(R1)*np.sin(np.mean(P1)*np.pi)
Z = (H*10000)/(1-H)
C.append(1/(2*np.pi*23.405138*np.imag(Z)))

################################################

filename = 'Csim_out_0.4v.csv'
data = np.genfromtxt(filename, delimiter=' ')

t1, v1, R1, P1 = data[:,0], data[:, 1], data[:, 2], data[:, 3]

R_m = ufloat(np.mean(R1), np.std(R1))
P_m = ufloat(np.mean(P1), np.std(P1))

H = np.mean(R1)*np.cos(np.mean(P1)*np.pi) + 1j*np.mean(R1)*np.sin(np.mean(P1)*np.pi)
Z = (H*10000)/(1-H)
C.append(1/(2*np.pi*23.405138*np.imag(Z)))

################################################


SNR = []
SNR = [-3.2493957646772498, -15.304125219585893, -17.55847698803266,
   -19.837388494926927, -23.53116735143342]


#SNR.append(1)
#SNR.append(1/4)
#SNR.append(0.8/4)
#SNR.append(0.6/4)
#SNR.append(0.4/4)


#No puse el valor de 0.2/4 porque me da un valor negativo de capacidad(encima 9e-6)
Cuf = [c*1000000 for c in C]
print("C = ",Cuf)
print("SNR = ",SNR)
plt.plot(SNR,Cuf,'b',marker="o")
plt.axhline(y=0.68-0.68*0.1, xmin=0, xmax=1,color = 'r',linestyle = '--')
plt.axhline(y=0.68+0.68*0.1, xmin=0, xmax=1,color = 'r',linestyle = '--')
plt.xlabel("SNR[dB]")
plt.ylabel("CAPACIDAD[μF]")
plt.legend(['N = 4000'])
plt.grid()
plt.show()
