import numpy as np
import matplotlib.pyplot as plt
from scipy import signal

from matplotlib import rcParams
rcParams['font.family'] = 'serif'


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
    S = np.abs(S)
    #S = S/np.max(S)
    return S

def fourierizar2(s):
    S = np.fft.fft(s)
    #S = np.split(S, 2)[0]
    #S = S/np.max(S)
    return S


if __name__ == '__main__':  #void main
    fs = 500
    Ts = 1/fs
    N = 4000
    ts = np.arange(0, N*Ts, Ts)
    fp = 50
    fm = 4
    rs = np.sin(2*np.pi*fp*ts)
    rp = np.sin(2*np.pi*fp*ts-np.pi/6)
    m = np.sin(2*np.pi*fm*ts)
    vs = np.multiply(rp, m)
    ns = np.random.normal(0, 1, N)
    
    fc = 6
    orden = 1000
    fss = medir_fs(ts)
    h, tau = filtro(fss, fc, orden)
    
    h2, tau2 = filtro(fss, fc, 50*orden)

    w, H = signal.freqz(h2)
    w = w/np.pi*fss/2
    H = np.abs(H)
    
    t, v, r, x, y, p, q, R, P = [], [], [], [], [], [], [], [], []
    vaux = []
    for i in range(2, N):    
        # Etapa de adquisición
        r.append(rs[i])
        v.append(vs[i])
        vaux.append(2*vs[i])
        # Etapa de procesamiento
        p, q = mezclar(r, vaux)
        x_out = filtrar(p, h)
        y_out = filtrar(q, h)
    
        R_out, P_out = np.hypot(y_out, x_out), np.arctan2(y_out, x_out)

        if ts[i] > 5*tau: 
            x.append(x_out)
            y.append(y_out)

            P.append(P_out)
            R.append(R_out)

            t.append(ts[i])
        
    fig, axs = plt.subplots(2,3)
    
    p = p[-len(t):]
    q = q[-len(t):]
    r = r[-len(t):]
    v = v[-len(t):]

    axs[0,0].plot(t, r)
    axs[0,0].set_title("r(t)")
    axs[1,0].plot(t, v)
    axs[1,0].set_title("v(t)")
    axs[0,1].plot(t, p)
    axs[0,1].set_title("p(t)")
    axs[1,1].plot(t, q)
    axs[1,1].set_title("q(t)")
    axs[0,2].plot(t, x)
    axs[0,2].set_title("x(t)")
    axs[1,2].plot(t, y)
    axs[1,2].set_title("y(t)")

    for axx in axs:
        for axy in axx:
            #axy.set_xlim([7, 8])
            axy.set_ylim([-2.2, 2.2])

    plt.show()
    
    fig, axs = plt.subplots(2,3)

    f = np.fft.fftfreq(len(t), 1/fs)
    #f = np.split(f, 2)[0]
    mx = np.max(fourierizar(r))
    axs[0,0].plot(f, fourierizar(r)/mx)
    axs[0,0].set_title("R(f)")
    axs[1,0].plot(f, fourierizar(v)/mx)
    axs[1,0].set_title("V(f)")
    axs[0,1].plot(w, H, ':k', -w, H, ':k', f, fourierizar(p)/mx)
    axs[0,1].set_title("P(f) = R(f)∗V(f)")
    axs[0,1].legend(["H(f)"])
    axs[1,1].plot(w, H, ':k', -w, H, ':k', f, fourierizar(q)/mx)
    axs[1,1].set_title("Q(f) = Ř(f)∗V(f)")
    axs[1,1].legend(["H(f)"])
    axs[0,2].plot(f, fourierizar(x)/mx)
    axs[0,2].set_title("X(f)=P(f)·H(f)")
    axs[1,2].plot(f, fourierizar(y)/mx)
    axs[1,2].set_title("Y(f)=Q(f)·H(f)")
    for axx in axs:
        for axy in axx:
            axy.set_xlim([-120, 120])
            axy.set_ylim([0, 1.1])
            #axy.grid(axis='x')
            axy.set_xticklabels([])
            axy.set_yticklabels([])
            axy.yaxis.set_ticks_position('none')
    
    plt.tight_layout()
    plt.show()