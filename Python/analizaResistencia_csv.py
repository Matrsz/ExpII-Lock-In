import numpy as np
from uncertainties import ufloat
import csv
import matplotlib.pyplot as plt
from scipy.signal import argrelextrema
from scipy import signal

f = 30 #no la uso igual

Resistencia = []

def fourierizar2(s):
    S = np.fft.fft(s)
    S = np.abs(S)
    S = S/np.max(S)
    return S

def pasabanda(fn, f1, f2, N):
    w1, w2 = f1/fn, f2/fn
    h_bp = signal.firwin(N, [w1, w2], pass_zero=False)
    h_bs = signal.firwin(N, [w1, w2])

    return [h_bp, h_bs]

def pasabajo(fn, fc, N):
    wc = fc/fn
    h_hp = signal.firwin(N, wc, pass_zero=False)
    h_lp = signal.firwin(N, wc)

    return [h_lp, h_hp]

def plot_filter(t, v, fn, h1, h2):
    Ts = (t[-1]-t[0])/len(t)
    f = np.fft.fftfreq(len(t), Ts)

    w, H1 = signal.freqz(h1)
    w, H2 = signal.freqz(h2)

    w = w/np.pi*fn
    fig, axs = plt.subplots(2,1)
    axs[0].plot(f, fourierizar2(v), 'b', w, np.abs(H1), ':k', -w, np.abs(H1), ':k')
    axs[0].set_title("Pasa Banda")
    axs[1].plot(f, fourierizar2(v), 'b', w, np.abs(H2), ':k', -w, np.abs(H2), ':k')
    axs[1].set_title("Rechaza Banda")
    plt.show()

def get_snr(v, h1, h2):
    s = signal.filtfilt(h1, [1], v)
    n = signal.filtfilt(h2, [1], v)

    s_rms = np.sqrt(np.mean(s**2))
    n_rms = np.sqrt(np.mean(n**2))

    print("Señal rms = ", s_rms)
    print("Ruido rms = ", n_rms)
    snr = s_rms/n_rms
    snr_db = 20*np.log10(snr)

    return snr_db


def analizar_snrs(filename, f0, plotting):
    data = np.genfromtxt(filename, delimiter=' ')
    
    Vp = data[0,9]
    t, v, R = data[:,0], data[:, 1]*Vp, data[:, 2]*Vp
    
    Ts = (t[-1]-t[0])/len(t)
    fs = 1/Ts
    fn = fs/2

    delta = 0.5
    N = 1001

    h_bp, h_bs = pasabanda(fn, f0-delta, f0+delta, N)

    if plotting:
        plot_filter(t, v, fn, h_bp, h_bs)

    snr_in = get_snr(v, h_bp, h_bs)
    print("SNR entrada = ", snr_in, " dB")

    fc = 0.5
    h_lp, h_hp = pasabajo(fn, fc, N)
    
    if plotting:
        plot_filter(t, R, fn, h_lp, h_hp)

    snr_out = get_snr(R, h_lp, h_hp)
    print("SNR salida = ", snr_out, " dB")

    return [snr_in, snr_out]


def analizar_resistencia(filename):
    data = np.genfromtxt(filename, delimiter=' ')
    R = 470

    t1, v1, R1, P1 = data[:,0], data[:, 1], data[:, 2], data[:, 3]
    
    H = np.mean(R1)*np.cos(np.mean(P1)*np.pi) + 1j*np.mean(R1)*np.sin(np.mean(P1)*np.pi)
    Z = (H*R)/(1-H)
    print(H)
    print(Z)
    R = np.abs(Z)
    return R

Resistencia = []
snrin = []

filenames = ['sim_out_4V4000.csv', 'sim_out_1V4000.csv', 'sim_out_0.8V4000.csv', 'sim_out_0.6V4000.csv', 'sim_out_0.4V4000.csv', 'sim_out_0.2V4000.csv']

for filename in filenames:
    Resistencia.append(analizar_resistencia(filename))
    snrin.append(analizar_snrs(filename, f, False)[0])

snrin[4] = snrin[4] - 4
Resistencia2 = []
snrin2 = []
filenames = ['sim_out_4V2000.csv', 'sim_out_1V2000.csv', 'sim_out_0.6V2000.csv', 'sim_out_0.8V2000.csv', 'sim_out_0.4V2000.csv', 'sim_out_0.2V2000.csv']

for filename in filenames:
    Resistencia2.append(analizar_resistencia(filename))
    #snrin2.append(analizar_snrs(filename, f, False)[0])


Resistencia3 = []
snrin3 = []
filenames = ['sim_out_1V1000.csv', 'sim_out_4V1000.csv', 'sim_out_0.8V1000.csv', 'sim_out_0.4V1000.csv', 'sim_out_0.6V1000.csv', 'sim_out_0.2V1000.csv']

for filename in filenames:
    Resistencia3.append(analizar_resistencia(filename))
    #snrin3.append(analizar_snrs(filename, f, False)[0])

#SNR = []
#
#SNR.append(1)
#SNR.append(1/4)
#SNR.append(0.8/4)
#SNR.append(0.6/4)
#SNR.append(0.4/4)
#SNR.append(0.2/4)
#
##SNR = [14.2, 3.2, 1.3, -1.5, -4.4, -15.5]
#SNR = [13.399485222497487, 2.9242719945388673, 0.6488520868543456, 
#-5.357905252717508, -8.113899133972234, -11.156481168356017]


plt.plot(snrin, Resistencia,marker="o", linestyle='None')
#plt.plot(snrin, Resistencia2,'g',marker="s", linestyle='None')
#plt.plot(snrin, Resistencia3,'r',marker="v", linestyle='None')

plt.axhline(y=470-23.5, xmin=0, xmax=1,color = 'k',linestyle = ':')
plt.axhline(y=470, xmin=0, xmax=1,color = 'k',linestyle = '--')
plt.axhline(y=470+23.5, xmin=0, xmax=1,color = 'k',linestyle = ':')
#plt.legend(['N = 4000', 'N = 2000', 'N = 1000'])
plt.gca().invert_xaxis()
plt.title("Divisor Resistivo, N = 4000")
plt.xlabel("SNR [dB]")
plt.ylabel("RESISTENCIA [Ω]")
plt.grid()
plt.tight_layout()
plt.show()

#REPORTAR RL, promedio ponderado:

# Resistencia:  [449.41819970479435, 461.1614183033961, 
# 468.39764731684977, 475.4430891343917, 518.8765566995003, 779.5540363354934]

#SNR:  [13.399485222497487, 2.9242719945388673, 0.6488520868543456,
#  -5.357905252717508, -8.113899133972234, -11.156481168356017]

Coeficientes = [0.4, 0.3, 0.2, 0.1]
RL = 0
for i in range(0,4,1):
    RL = RL + Resistencia[i]*Coeficientes[i]
    print(i)
Incerteza = (Resistencia[3] - Resistencia[0])/2
print("RL: ",RL, " ± ",Incerteza)
print("Entonces reportare (460±20)Ω => incerteza relativa de 4.3%")

