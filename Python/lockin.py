#Reads an A/D Input Channel
from __future__ import absolute_import, division, print_function
from builtins import *
from cmath import pi, tau
from pickle import TRUE
from time import sleep  # @UnusedWildImport
from mcculw import ul
from mcculw.device_info import DaqDeviceInfo
from scipy import signal
import matplotlib.pyplot as plt
import numpy as np
import time

try:
    from console_examples_util import config_first_detected_device
except ImportError:
    from .console_examples_util import config_first_detected_device

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
        ai_range = ai_info.supported_ranges[0]
        ao_info = daq_dev_info.get_ao_info()
        ao_range = ao_info.supported_ranges[0]

        return [board_num, ai_range, ao_range]

    except Exception as e:  #En caso de error
        print('\nError\n', e)

    finally:
        ul.release_daq_device(board_num)

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

# Mide la frecuencia de de muestreo efectiva del sistema de medición
# Considerando el tiempo de lectura y escritura a los canales del instrumento
# Se considera despreciable el tiempo de CPU respecto al tiempo de I/O
def medir_fs(board_num, ai_range, ao_range, in_channels, out_channels):
    start = time.time()
    for c in in_channels:
        ul.v_in(board_num, c, ai_range)
    for c in out_channels:
        ul.v_out(board_num, c, ao_range, 0)
    Ts = time.time()-start
    return 1/Ts

# Retorna como máximo los últimos N elementos de los vectores de entrada
def limpiar_vectores(xs, N):
    return [x[-N:] if len(x) >= N else x for x in xs]

if __name__ == '__main__':  #void main
    board_num, ai_range, ao_range = inicializar()
    in_channels = [0, 1]
    out_channels = [2, 3]

    fs = medir_fs(board_num, ai_range, ao_range, in_channels, out_channels)
    fc = 50
    N = 100
    h, tau = filtro(fs, fc, N)
    print("Tau =", tau)
    tau_flag = True

    t, v, r, R, P = [], [], [], [], []
    max_muestras = 1000
    start = time.time()

    while True:
        inicio = time.time()-start

        # Etapa de adquisición
        r_in, v_in = adquisicion(board_num, ai_range, in_channels)
        r.append(r_in)
        v.append(v_in)

        # Etapa de procesamiento
        p, q = mezclar(r, v)
        x, y = filtrar(p, h), filtrar(q, h)
        R_out, P_out = np.hypot(y, x), np.arctan2(y, x)

        P.append(P_out)
        R.append(R_out)

        # Etapa de escritura
        escritura(board_num, ao_range, out_channels, [R_out, P_out])
        
        fin = time.time()-start
        t_now = (inicio+fin)/2
        if t_now >= tau and tau_flag:
            print("Tau superado, Mediciones válidas.")
            tau_flag = False

        t.append(t_now)

        t, v, r, R, P = limpiar_vectores([t, v, r, R, P], max_muestras)




