import numpy as np
import csv
import matplotlib.pyplot as plt
from scipy import signal

def fourierizar2(s):
    S = np.fft.fft(s)
    S = np.abs(S)
    S = S/np.max(S)
    return S


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
    f1, f2 = f0-delta, f0+delta
    w1, w2 = f1/fn, f2/fn
    h_bp = signal.firwin(N, [w1, w2], pass_zero=False)
    h_bs = signal.firwin(N, [w1, w2])

    w, H_bp = signal.freqz(h_bp)
    w, H_bs = signal.freqz(h_bs)

    fig, axs = plt.subplots(2,1)
    axs[0].plot(w/np.pi*fn, np.abs(H_bp), f, fourierizar2(v))
    axs[0].set_xlim([0, max(f)])
    axs[0].set_title("Pasa Banda")
    axs[1].plot(w/np.pi*fn, np.abs(H_bs), f, fourierizar2(v))
    axs[1].set_xlim([0, max(f)])
    axs[1].set_title("Rechaza Banda")
    plt.show()

    s = signal.filtfilt(h_bp, [1], v)
    n = signal.filtfilt(h_bs, [1], v)

    s_rms = np.sqrt(np.mean(s**2))
    n_rms = np.sqrt(np.mean(n**2))

    print("Señal rms = ", s_rms)
    print("Ruido rms = ", n_rms)
    snr = s_rms/n_rms
    snr_db_in = 10*np.log10(snr)
    print("SNR entrada = ", snr_db_in, " dB")

    #fig, axs = plt.subplots(2,1)
    #axs[0].plot(t, s)
    #axs[0].set_title("Señal")
    #axs[1].plot(t, n)
    #axs[1].set_title("Ruido")
    #plt.show()

    fc = 1
    wc = fc/fn
    h_hp = signal.firwin(N, wc, pass_zero=False)
    h_lp = signal.firwin(N, wc)

    w, H_lp = signal.freqz(h_lp)
    w, H_hp = signal.freqz(h_hp)

    fig, axs = plt.subplots(2,1)
    axs[0].plot(w/np.pi*fn, np.abs(H_lp), f, fourierizar2(P))
    axs[0].set_xlim([-0.1, max(f)])
    axs[0].set_title("Pasa Banda")
    axs[1].plot(w/np.pi*fn, np.abs(H_hp), f, fourierizar2(P))
    axs[1].set_xlim([-0.1, max(f)])
    axs[1].set_title("Rechaza Banda")
    plt.show()

    s = signal.filtfilt(h_lp, [1], P)
    n = signal.filtfilt(h_hp, [1], P)

    #fig, axs = plt.subplots(2,1)
    #axs[0].plot(t, s)
    #axs[0].set_title("Señal")
    #axs[1].plot(t, n)
    #axs[1].set_title("Ruido")
    #plt.show()

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