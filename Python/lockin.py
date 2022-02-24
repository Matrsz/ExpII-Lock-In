from __future__ import absolute_import, division, print_function
from builtins import *
from pickle import TRUE
from telnetlib import TSPEED
from mcculw import ul
from mcculw.device_info import DaqDeviceInfo
from scipy import signal
import matplotlib.pyplot as plt
import numpy as np
import time
import csv
from console_examples_util import config_first_detected_device

# Inicializa la placa de adquisición
def inicializar():
    dev_id_list = []
    board_num = 0
    try:
        config_first_detected_device(board_num, dev_id_list)     
        daq_dev_info = DaqDeviceInfo(board_num)

        print('\nActive DAQ device: ', daq_dev_info.product_name, ' (',
              daq_dev_info.unique_id, ')\n', sep='')

        ai_info = daq_dev_info.get_ai_info()
        ai_range = ai_info.supported_ranges[0] # el 2 esta en 5V
        ao_info = daq_dev_info.get_ao_info()
        ao_range = ao_info.supported_ranges[0]
        return [board_num, ai_range, ao_range]

    except Exception as e:  #En caso de error
        print('\nError\n', e)

# Lee de la placa de adquisición todos los valores en los canales channels y los retorna
def adquisicion(board_num, ai_range, channels):
    return [ul.v_in(board_num, c, ai_range) for c in channels]

# Escribe sobre los canales channels los valores de índice correspondiente en values
def escritura(board_num, ao_range, channels, values):
    for c, v in zip(channels, values):
        ul.v_out(board_num, c, ao_range, v)
    return

# Obtiene la señal de referencia original y desfasada usando la transformada de Hilbert
# Retorna el producto de la entrada por la referencia original y la referencia desfasada
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

# Mide la frecuencia de de muestreo efectiva del sistema de medición
# Considerando el tiempo de lectura y escritura a los canales del instrumento
# Se considera despreciable el tiempo de CPU respecto al tiempo de I/O
def medir_Ts(board_num, ai_range, ao_range, in_channels, out_channels, test_time):
    Ts = []
    Vn = []
    test_start = time.time()
    while time.time()-test_start < test_time:
        start = time.time()
        Vn.append(ul.v_in(board_num, 2, ai_range))
        ul.v_in(board_num, 3, ai_range)
        for c in out_channels:
            ul.v_out(board_num, c, ao_range, 0)
        Ts.append(time.time()-start)
    Vp = np.sqrt(np.mean(np.power(Vn,2))) * np.sqrt(2)
    return [np.mean(Ts), np.std(Ts), Vp]

# Retorna como máximo los últimos N elementos de los vectores de entrada
def limpiar_vectores(xs, N):
    return [x[-N:] if len(x) >= N else x for x in xs]

if __name__ == '__main__':  #void main
    realtime = False
    filename = 'sim_out.csv'

    #Inicialización de la placa de adquisición, medición del tiempo de muestreo
    board_num, ai_range, ao_range = inicializar()
    in_channels = [2, 3]
    out_channels = [0, 1] if realtime else [] # 13 y 14

    Ts, Ts_err, Vp = medir_Ts(board_num, ai_range, ao_range, in_channels, out_channels, 5)
    print("VP:",Vp)
    print("Ts = ", Ts, " +/- ", Ts_err, "s")

    #Inicialización del filtro pasa bajos para la frecuencia de muestreo medida
    fs = 1/Ts 
    fc = 0.01    
    N = 4000 #debe ser <= a max_muestras
    h, tau = filtro(fs, fc, N)
    print("Tau = ", tau, " s")

    tau_flag = True
    max_muestras = 5000 #debe ser >= al orden del filtro N
    v, r = [], []
    t_now = 0
    start = time.time()

    file = open(filename, 'w', newline='')
    writer = csv.writer(file, delimiter=' ', quotechar='|', quoting=csv.QUOTE_MINIMAL)

    while t_now < 15*tau:
        inicio = time.time()-start

        # Etapa de adquisición
        r_in, v_in = adquisicion(board_num, ai_range, in_channels)
        r_in = 2*r_in
        r.append(r_in)
        v.append(v_in)

        # Etapa de procesamiento
        p, q = mezclar(r, v)
        x, y = filtrar(p, h)/(Vp*Vp), filtrar(q, h)/(Vp*Vp)
        R_out, P_out = np.hypot(y, x), np.arctan2(y, x)/np.pi

        # Etapa de escritura (a placa de adquisición)
        if realtime:
            escritura(board_num, ao_range, out_channels, [R_out, P_out])
        
        fin = time.time()-start        
        t_now = (inicio+fin)/2
        
        if t_now > 8*tau:
            if tau_flag:
                print("5 tau superado, Mediciones válidas.")
                tau_flag = False
            writer.writerow([t_now, v_in/Vp, R_out, P_out, r_in, x, y, p[-1], q[-1], Vp, fs, tau, N, fc])
        
        v, r = limpiar_vectores([v, r], max_muestras)


    file.close()

#    plt.plot(r)
#    plt.plot(v)
#    plt.show()

##    if len(t)>len(R):
##        t.pop(0)
##    H = np.mean(R)*np.cos(np.mean(P)) + 1j*np.mean(R)*np.sin(np.mean(P))
##    Z = H / (1-H) * 1.0206e6
##    print("Resistencia:",np.real(Z),"\nCapacitancia:",1/(2*pi*25*np.imag(Z)))
##    print("H:",H)
##
##
##    fig, axs = plt.subplots(2,1)
##    axs[0].plot(t,R, "b")
##    axs[0].set_title("Amplitud")
##    axs[0].set_ylim([0, np.max(R)])
##    axs[1].plot(t, P, "r")
##    axs[1].set_ylim([-1, 1])
##    axs[1].set_title("Fase")
##    plt.show()


    ul.release_daq_device(board_num)
        

