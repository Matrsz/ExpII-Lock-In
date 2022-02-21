import numpy as np
from uncertainties import ufloat
import csv
import matplotlib.pyplot as plt

filename = 'sim_out.csv'
data = np.genfromtxt(filename, delimiter=' ')

t, R, P = data[:,0], data[:, 1], data[:, 2]

plt.plot(t, R, t, P)
plt.show()

R_m = ufloat(np.mean(R), np.std(R))
P_m = ufloat(np.mean(P), np.std(P))

print("R = ", R_m)
print("P = ", P_m)