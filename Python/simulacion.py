import numpy as np
import matplotlib.pyplot as plt
from scipy import signal


# Si Nx >= Nh toma los últimos Nh elementos de x, si Nx < Nh completa x con 0s
# Aplica el fir definido por h, retornando y[n] = el producto interno de h con x 
def mezclar(r, v):
    aux = signal.hilbert(r)
    p = np.multiply(v, np.real(aux))
    q = np.multiply(v, np.imag(aux))
    return [p, q]

# Si Nx >= Nh toma los últimos Nh elementos de x, si Nx < Nh completa x con 0s
# Aplica el fir definido por h, retornando y[n] = el producto interno de h con x 
def filtrar(x, h):
    Nh, Nx = len(h), len(x)
    x = x[-Nh:] if Nx >= Nh else np.pad(x, (Nh-Nx, 0))
    return np.dot(h, np.flip(x))

# Diseña un FIR de frecuenca de corte fc y orden N para frecuencia de muestreo fs
def filtro(fs, fc, N):
    w = fc/(fs/2)
    h = signal.firwin(N, w, window = "hamming")
    tau = (N-1)/(2*fs)
    return [h, tau]

# Retorna como máximo los últimos N elementos de los vectores de entrada
def limpiar_vectores(xs, N):
    return [x[-N:] if len(x) >= N else x for x in xs]


# Inicializa el sistema de medición, midiendo la frecuencia de de muestreo efectiva
# Considerando el tiempo de lectura y escritura a los canales del instrumento
def medir_fs(t):
    Ts = t[1]-t[0]
    return 1/Ts

def fourierizar(s):
    S = np.fft.fft(s)
    #S = np.split(S, 2)[0]
    #S = np.abs(S)
    #S = S/np.max(S)
    return S


if __name__ == '__main__':  #void main
    fs = 500
    Ts = 1/fs
    N = 1000
    ts = np.arange(0, N*Ts, Ts)
    fp = 10
    fm = 5
    rs = np.cos(2*np.pi*fp*ts)
    #m = np.sin(2*np.pi*fm*ts)+1
    #m = np.ones(len(rs))
    #vs = np.sin(2*np.pi*fp*ts)
    vs = np.cos(2*np.pi*fp*ts)
    #vs = np.cos(2*np.pi*fp*ts - np.pi/4)
    ns = np.random.normal(0, 1, N)
    
    fc = 15
    orden = 200
    fss = medir_fs(ts)
    h, tau = filtro(fss, fc, orden)
    
    w, H = signal.freqz(h)
    plt.plot(w/np.pi*fss/2, np.abs(H))
    plt.show()
    
    t, v, r, x, y, p, q, R, P = [], [], [], [], [], [], [], [], []
    
    for i in range(2, N):    
        # Etapa de adquisición
        r.append(rs[i])
        v.append(2*vs[i])
    
        # Etapa de procesamiento
        p, q = mezclar(r, v)
        x_out = filtrar(p, h)
        y_out = filtrar(q, h)
        x.append(x_out)
        y.append(y_out)
    
        R_out, P_out = np.hypot(y_out, x_out), np.arctan2(y_out, x_out)
        P.append(P_out)
        R.append(R_out)
    
        t.append(ts[i])
        
    fig, axs = plt.subplots(2,3)
    
    axs[0,0].plot(t, r)
    axs[0,0].set_title("Referencia")
    axs[1,0].plot(t, v)
    axs[1,0].set_title("Señal")
    axs[0,1].plot(t, p, t, q)
    axs[0,1].set_title("Señal Mezclada")
    axs[1,1].plot(t, x, t, y)
    axs[1,1].set_title("Señal Filtrada")
    axs[0,2].plot(t, R)
    axs[0,2].plot([tau, tau], [min(R), max(R)])
    axs[0,2].set_title("Amplitud lock in")
    axs[1,2].plot(t, P)
    axs[1,2].plot([tau, tau], [min(P), max(P)])
    axs[1,2].set_title("Fase lock in")

    plt.show()
    
    fig, axs = plt.subplots(2,3)

    f = np.fft.fftfreq(len(t), 1/fs)
    #f = np.split(f, 2)[0]

    axs[0,0].plot(f, np.real(fourierizar(r)), f, np.imag(fourierizar(r)))
    axs[0,0].set_title("Referencia")
    axs[1,0].plot(f, np.real(fourierizar(v)), f, np.imag(fourierizar(v)))
    axs[1,0].set_title("Señal")
    axs[0,1].plot(f, np.real(fourierizar(p)), f, np.imag(fourierizar(p)))
    axs[0,1].set_title("Señal Mezclada p")
    axs[1,1].plot(f, np.real(fourierizar(q)), f, np.imag(fourierizar(q)))
    axs[1,1].set_title("Señal Mezclada q")
    axs[0,2].plot(f, np.real(fourierizar(x)), f, np.imag(fourierizar(x)))
    axs[0,2].set_title("Señal Filtrada x")
    axs[1,2].plot(f, np.real(fourierizar(y)), f, np.imag(fourierizar(y)))
    axs[1,2].set_title("Señal Filtrada y")
#    axs[1,2].plot(f, P)
#    axs[1,2].set_title("Fase lock in")

    plt.show()