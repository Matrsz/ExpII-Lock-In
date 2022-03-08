import numpy as np
import uncertainties as uc
import uncertainties.umath as um
import csv
import matplotlib.pyplot as plt

#filename = 'Csim_out_4V.csv'
filename = 'sim_out_4V4000.csv'

data = np.genfromtxt(filename, delimiter=' ')

t, v, R, P = data[:,0], data[:, 1], data[:, 2], data[:, 3]


xlims = [24, 25]
plt.subplot(1, 2, 1)
plt.plot(t, v)
plt.title("Entrada Lockin")
plt.xlim(xlims)
plt.subplot(2, 2, 2)
plt.plot(t, R, "b")
plt.xlim(xlims)
plt.title("Amplitud")
plt.ylim([0, np.max(R)*1.2])
plt.subplot(2,2,4)
plt.plot(t, P*np.pi, "r")
plt.xlim(xlims)
plt.ylim([-1, 1])
plt.title("Fase")
plt.tight_layout()
#plt.show()

def fourierizar(s):
    S = np.fft.fft(s)
    S = np.abs(S)
    return S

fig, axs = plt.subplots(2,2)
Ts = (t[-1]-t[0])/len(t)
f = np.fft.fftfreq(len(t), Ts)
lims = [np.min(v)*1.1, np.max(v)*1.1]
axs[0,0].plot(t, v)
axs[0,0].set_title("Entrada Normalizada")
axs[0,0].set_xlim(xlims)
axs[0,0].set_ylim(lims)
axs[0,1].plot(t, R)
axs[0,1].set_ylim(lims)
axs[0,1].set_title("Salida Normalizada")
axs[0,1].set_xlim(xlims)
axs[1,0].plot(f, fourierizar(v))
axs[1,0].set_ylim([0, np.max(fourierizar(v))*1.1])
axs[1,1].plot(f, fourierizar(R))
axs[1,1].set_ylim([0, np.max(fourierizar(R))*1.1])
axs[0,0].set_xlabel("t [s]")
axs[0,1].set_xlabel("t [s]")
axs[0,0].set_yticks([-1, -0.5, 0, 0.5, 1])
axs[0,1].set_yticks([-1, -0.5, 0, 0.5, 1])
axs[1,0].set_xlabel("f [Hz]")
axs[1,1].set_xlabel("f [Hz]")
axs[1,0].set_xticks([-100, -50, 0, 50, 100])
axs[1,1].set_xticks([-100, -50, 0, 50, 100])

for ax in axs:
    for ay in ax:
        ay.grid()
plt.tight_layout()
#plt.show()


R = uc.ufloat(np.mean(R), np.std(R))
P = uc.ufloat(np.mean(P), np.std(P))

H_re = R*um.cos(P*np.pi)
H_im = R*um.sin(P*np.pi)

R_s = uc.ufloat(470, 23.5)

a = H_re
b = H_im
c = 1-H_re
d = -H_im

R = (a*c+b*d)/(c**2+d**2) * R_s 
X = (b*c-a*d)/(c**2+d**2) * R_s

C = 1/(2*np.pi*23.405138*X)*1e6

print(H_re, " + j", H_im)

print(R)
print(C)
#H = R_m*np.cos(P_m) + 1j*R_m*np.sin(P_m)
#H = np.mean(R)*np.cos(np.mean(P)*np.pi) + 1j*np.mean(R)*np.sin(np.mean(P)*np.pi)
#Z = (H*10000)/(1-H)
#print("Z = ", Z)
#print("Resistencia:",np.real(Z),"\nCapacitancia:",-1/(2*np.pi*23.405138*np.imag(Z)))
#print("H:",H)j*R*