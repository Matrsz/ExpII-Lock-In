import numpy as np
import uncertainties as uc
import uncertainties.umath as um
import csv
import matplotlib.pyplot as plt
from scipy.signal import argrelextrema
from scipy import signal

def analizar_resistencia(filename):
    data = np.genfromtxt(filename, delimiter=' ')

    t1, v1, R, P = data[:,0], data[:, 1], data[:, 2], data[:, 3]

    R = uc.ufloat(np.mean(R), np.std(R))
    P = uc.ufloat(np.mean(P), np.std(P))/2

    H_re = R*um.cos(P*np.pi)
    H_im = R*um.sin(P*np.pi)

    R_s = uc.ufloat(470, 20)

    a = H_re
    b = H_im
    c = 1-H_re
    d = -H_im

    R = (a*c+b*d)/(c**2+d**2) * R_s 
    print(R)

    return R


filenames = ['sim_out_0.2V1000.csv', 'sim_out_0.2V2000.csv', 'sim_out_0.2V4000.csv', 'sim_out_0.2V8000.csv', 'sim_out_0.2V16000.csv']

r_m = []
r_err = []
for filename in filenames:
    r = analizar_resistencia(filename)
    r_m.append(r.nominal_value)
    r_err.append(r.std_dev)
    
N =[1000,2000,4000,8000,16000]
r_err[0] = r_err[0]*2
plt.errorbar(N, r_m, r_err ,marker="p", capsize=4)
plt.axhline(y=470-23.5, xmin=0, xmax=1,color = 'k',linestyle = ':')
plt.axhline(y=470, xmin=0, xmax=1,color = 'k',linestyle = '--')
plt.axhline(y=470+23.5, xmin=0, xmax=1,color = 'k',linestyle = ':')
plt.xlabel("N")
plt.title("Resistencia Medida a SNR = -22.5 dB")
plt.ylabel("RESISTENCIA [Ω]")
plt.grid()
plt.tight_layout()
plt.show()
