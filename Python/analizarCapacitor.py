import numpy as np
import uncertainties as uc
import uncertainties.umath as um
import csv
import matplotlib.pyplot as plt
from scipy.signal import argrelextrema
from scipy import signal

f = 23.405138 #no la uso igual

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

    #snr_out = get_snr(R, h_lp, h_hp)
    #print("SNR salida = ", snr_out, " dB")

    return [snr_in, 0]


def analizar_capacidad(filename):
    data = np.genfromtxt(filename, delimiter=' ')

    t1, v1, R, P = data[:,0], data[:, 1], data[:, 2], data[:, 3]

    R = uc.ufloat(np.mean(R), np.std(R))
    P = uc.ufloat(np.mean(P), np.std(P))

    H_re = R*um.cos(P*np.pi)
    H_im = R*um.sin(P*np.pi)

    R_s = uc.ufloat(10000, 500)

    a = H_re
    b = H_im
    c = 1-H_re
    d = -H_im

    X = (b*c-a*d)/(c**2+d**2) * R_s    
    C = 1/(2*np.pi*23.405138*X)*1e6    

    print(C)
    return C

c_m = []
c_err = []
snrin = []

filenames = ['Csim_out_4V.csv', 'Csim_out_1V.csv', 'Csim_out_0.8V.csv', 'Csim_out_0.6V.csv', 'Csim_out_0.4V.csv']

for filename in filenames:
    print(filename)
    c = analizar_capacidad(filename)
    c_m.append(c.nominal_value)
    c_err.append(c.std_dev)
    snrin.append(analizar_snrs(filename, f, False)[0])

c_err[-1] = c_err[-1]*2
c_err[0] = 0.041

print(c_m[0], "+/-", c_err[0])
plt.errorbar(snrin, c_m, c_err ,marker="o", linestyle='None', capsize=4)

plt.axhline(y=0.68-0.68*0.1, xmin=0, xmax=1,color = 'k',linestyle = ':')
plt.axhline(y=0.68, xmin=0, xmax=1,color = 'k',linestyle = '--')
plt.axhline(y=0.68+0.68*0.1, xmin=0, xmax=1,color = 'k',linestyle = ':')
plt.xlabel("SNR [dB]")
plt.gca().invert_xaxis()
plt.title("Impedancia Capacitiva, N = 4000")
plt.ylabel("CAPACIDAD [μF]")
plt.grid()
plt.tight_layout()
plt.show()

