import numpy as np
import csv
import matplotlib.pyplot as plt
from scipy import signal

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


def get_snrs(filename, plotting):
    data = np.genfromtxt(filename, delimiter=' ')
    
    Vp = data[0,9]
    t, v, R = data[:,0], data[:, 1]*Vp, data[:, 2]*Vp
    
    Ts = (t[-1]-t[0])/len(t)
    fs = 1/Ts
    fn = fs/2

    f0 = 30
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

labels = ['4V', '1V', '0.8V', '0.6V', '0.4V', '0.2V']
filenames = ['sim_out_'+x+'4000.csv' for x in labels]

snrins = []
snrouts = []
snrgain = []
for filename in filenames:
    snr_in, snr_out = get_snrs(filename, False)
    snrins.append(snr_in)
    snrouts.append(snr_out)
    snrgain.append(snr_out-snr_in)
    print("Ganancia en SNR = ", snr_out - snr_in, " dB")

plt.plot(snrins)
plt.plot(snrouts)
plt.show()