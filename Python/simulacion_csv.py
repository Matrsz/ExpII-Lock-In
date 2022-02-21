import numpy as np
from scipy import signal
import csv

# Si Nx >= Nh toma los últimos Nh elementos de x, si Nx < Nh completa x con 0s
# Aplica el fir definido por h, retornando y[n] = el producto interno de h con x 
def mezclar(r, v):
    aux = signal.hilbert(r)
    p = np.multiply(v, np.real(aux))
    q = np.multiply(v, np.real(aux))
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

if __name__ == '__main__':  #void main
    filename = 'sim_out.csv'
    fs = 500
    Ts = 1/fs
    N = 1000
    ts = np.arange(0, N*Ts, Ts)
    fp = 50
    fm = 10
    rs = np.sin(2*np.pi*fp*ts)
    m = np.sin(2*np.pi*fm*ts)+1
    vs = np.multiply(rs, m+1)
    ns = np.random.normal(0, 1, N)

    file = open('sim_out.csv', 'w', newline='')
    writer = csv.writer(file, delimiter=' ', quotechar='|', quoting=csv.QUOTE_MINIMAL)

    fc = 15
    orden = 200

    fss = medir_fs(ts)
    h, tau = filtro(fss, fc, orden)
    
    v, r = [], []
    
    for i in range(2, N):    
        # Etapa de adquisición
        r.append(rs[i])
        v.append(vs[i]+ns[i])
    
        # Etapa de procesamiento
        p, q = mezclar(r, v)
        x = filtrar(p, h)
        y = filtrar(q, h)
    
        R_out, P_out = np.hypot(y, x), np.arctan2(y, x)
        t = ts[i]

        writer.writerow([t, P_out, R_out])

    ## Q = np.abs(np.fft.fft(qs))
    ## F = np.fft.fftfreq(len(ts), Tss)
    ## Q = np.split(Q, 2)[0]
    ## Q = np.abs(Q)
    ## Q = Q/np.max(Q)
    ## F = np.split(F, 2)[0]
    ## plt.plot(F, Q)
    ## w, H = signal.freqz(h)
    ## plt.plot(w/np.pi*fss/2, np.abs(H))
    ## plt.show()