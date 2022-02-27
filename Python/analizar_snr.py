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

    fig, axs = plt.subplots(2,1)
    axs[0].plot(w/np.pi*fn, np.abs(H1), f, fourierizar2(v))
    axs[0].set_title("Pasa Banda")
    axs[1].plot(w/np.pi*fn, np.abs(H2), f, fourierizar2(v))
    axs[1].set_title("Rechaza Banda")
    plt.show()


def get_snrs(filename):
    data = np.genfromtxt(filename, delimiter=' ')

    t, v, R, P = data[:,0], data[:, 1], data[:, 2], data[:, 3]

    Ts = (t[-1]-t[0])/len(t)
    f = np.fft.fftfreq(len(t), Ts)

    fs = 1/Ts
    fn = fs/2

    f0 = 30
    delta = 1
    N = 1001

    h_bp, h_bs = pasabanda(fn, f0-delta, f0+delta, N)
    plot_filter(t, v, fn, h_bp, h_bs)

    s = signal.filtfilt(h_bp, [1], v)
    n = signal.filtfilt(h_bs, [1], v)

    s_rms = np.sqrt(np.mean(s**2))
    n_rms = np.sqrt(np.mean(n**2))

    print("Señal rms = ", s_rms)
    print("Ruido rms = ", n_rms)
    snr = s_rms/n_rms
    snr_db_in = 10*np.log10(snr)
    print("SNR entrada = ", snr_db_in, " dB")

    fc = 1
    h_lp, h_hp = pasabajo(fn, fc, N)
    plot_filter(t, P, fn, h_lp, h_hp)

    s = signal.filtfilt(h_lp, [1], P)
    n = signal.filtfilt(h_hp, [1], P)

    s_rms = np.sqrt(np.mean(s**2))
    n_rms = np.sqrt(np.mean(n**2))

    print("Señal rms = ", s_rms)
    print("Ruido rms = ", n_rms)
    snr = s_rms/n_rms

    snr_db_out = 10*np.log10(snr)
    print("SNR salida = ", snr_db_out, " dB")

    return [snr_db_in, snr_db_out]

labels = ['4V', '1V', '0.8V', '0.6V', '0.4V', '0.2V']
filenames = ['sim_out_'+x+'4000.csv' for x in labels]

snrins = []
snrouts = []
for filename in filenames:
    snr_in, snr_out = get_snrs(filename)
    snrins.append(snr_in)
    snrouts.append(snr_out)
    print("Ganancia en SNR = ", snr_out - snr_in, " dB")

print(snrins)