from cmath import phase
import numpy as np 
import matplotlib.pyplot as plt

Rs = 1000e3
R = 100e3
C = 33e-9

f = np.linspace(0, 200, 1000)
w = 2*np.pi*f

print(w)

H = (R+1/(1j*w*C))/(R+Rs+1/(1j*w*C))

fig, axs = plt.subplots(2,1)
axs[0].plot(f, np.abs(H))
axs[0].set_title("Amplitud")
axs[1].plot(f, np.angle(H)/np.pi*180)
axs[1].set_title("Fase")
plt.show()
