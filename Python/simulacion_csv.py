import numpy as np
from scipy import signal
import csv

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
def medir_Ts(t, r, test_time):
    Ts = []
    Vn = []
    i = 1
    while t[i] < test_time:
        Vn.append(r[i])
        Ts.append(t[i]-t[i-1])
        i += 1
    Vp = np.sqrt(np.mean(np.power(Vn,2))) * np.sqrt(2)
    return [np.mean(Ts), np.std(Ts), Vp]

def limpiar_vectores(xs, N):
    return [x[-N:] if len(x) >= N else x for x in xs]

if __name__ == '__main__':  #void main
    filename = 'sim_out.csv'
    fs = 500
    Ts = 1/fs
    N = 40000
    ts = np.arange(0, N*Ts, Ts)
    
    fp = 5
    rs = np.sin(2*np.pi*fp*ts)
    ns = np.random.normal(0, 1, N)*2
    vs = np.sin(2*np.pi*fp*ts)/2 + ns


    file = open('sim_out.csv', 'w', newline='')
    writer = csv.writer(file, delimiter=' ', quotechar='|', quoting=csv.QUOTE_MINIMAL)

    fc = 0.001
    orden = 4000
    Tss, Ts_err, Vp = medir_Ts(ts, rs, 10)
    print(Tss, Vp)
    h, tau = filtro(1/Tss, fc, orden)

    print("Tau = ", tau)
    
    v, r = [], []
    i = 0
    t_now = 0
    
    max_muestras = 5000 #debe ser >= al orden del filtro N
    
    while True:
        # Etapa de adquisición
        r_in = 2*rs[i]
        r.append(r_in)
        v_in = vs[i]+ns[i]
        v.append(v_in)
    
        # Etapa de procesamiento
        p, q = mezclar(r, v)
        x = filtrar(p, h)/(Vp*Vp)
        y = filtrar(q, h)/(Vp*Vp)
    
        R_out, P_out = np.hypot(y, x), np.arctan2(y, x)

        t_now = ts[i]
        if t_now > 5*tau:
            writer.writerow([t_now, v_in, R_out, P_out])
        if t_now >= 10*tau:
            break
        
        i += 1        
        v, r = limpiar_vectores([v, r], max_muestras)
