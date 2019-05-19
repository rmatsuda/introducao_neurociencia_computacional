import scipy as sp
import pylab as plt
from scipy.integrate import odeint, ode
from scipy.signal import find_peaks
import numpy as np
from scipy import stats
import scipy.linalg as lin

## Full Hodgkin-Huxley Model (copied from Computational Lab 2)

# Constants
C_m = 1.0  # membrane capacitance, in uF/cm^2
g_Na = 120.0  # maximum conducances, in mS/cm^2
g_K = 36.0
g_L = 0.3
E_Na = 50.0  # Nernst reversal potentials, in mV
E_K = -77.0
E_L = -54.4


# Channel gating kinetics
# Functions of membrane voltage
def alpha_m(V): return 0.1 * (V + 40.0) / (1.0 - sp.exp(-(V + 40.0) / 10.0))


def beta_m(V):  return 4.0 * sp.exp(-(V + 65.0) / 18.0)


def alpha_h(V): return 0.07 * sp.exp(-(V + 65.0) / 20.0)


def beta_h(V):  return 1.0 / (1.0 + sp.exp(-(V + 35.0) / 10.0))


def alpha_n(V): return 0.01 * (V + 55.0) / (1.0 - sp.exp(-(V + 55.0) / 10.0))


def beta_n(V):  return 0.125 * sp.exp(-(V + 65) / 80.0)


# Membrane currents (in uA/cm^2)
#  Sodium (Na = element name)
def I_Na(V, m, h): return g_Na * m ** 3 * h * (V - E_Na)


#  Potassium (K = element name)
def I_K(V, n):  return g_K * n ** 4 * (V - E_K)


#  Leak
def I_L(V):     return g_L * (V - E_L)


# External current
voltage = 8
start=0
finish=1
# L=13.60 # limiar A=10
L=15#mesma amplitude A=10
#L=16.8 # limiar A=7
L=17.5#mesma amplitude A=7
A=7
def I_inj(t):  # step up 10 uA/cm^2 every 100ms for 400ms
    return voltage * (t >= start) - voltage * (t > finish) + A * (t > finish+L) - A * (t > finish+L+1)
    # return 10*t

# The time to integrate over
dt=0.05
t = sp.arange(0.0, 50.0, dt)


plt.figure()

plt.plot(t, I_inj(t),label='corrente')
plt.xlabel('t (ms)')
plt.ylabel('$I_{inj}$ ($\\mu{A}/cm^2$)')
#plt.legend(voltage)
#plt.show()

def dALLdt(X, t):
    V, m, h, n = X

    # calculate membrane potential & activation variables
    dVdt = (I_inj(t) - I_Na(V, m, h) - I_K(V, n) - I_L(V)) / C_m
    dmdt = alpha_m(V) * (1.0 - m) - beta_m(V) * m
    dhdt = alpha_h(V) * (1.0 - h) - beta_h(V) * h
    dndt = alpha_n(V) * (1.0 - n) - beta_n(V) * n
    return dVdt, dmdt, dhdt, dndt

#V0 = -65 mV, m(-65) = 0,05,  h(-65) = 0,6, n(-65) = 0,32

X = odeint(dALLdt, [-65, 0.05, 0.6, 0.32], t)
V = X[:, 0]

#plt.figure()

plt.title('Hodgkin-Huxley Neuron')
plt.plot(t, V, 'k', label='potencial')
plt.ylabel('V (mV)')
plt.legend()
plt.show()

peaks, _ = find_peaks(V, height=0, threshold=0)
print('Amplitude dos picos:',V[peaks], 'mV')
# zero_crossings = np.where(np.diff(np.sign(V)))[0]
# if len(zero_crossings)>0:
#     print('Voltage', voltage[i], 'Latencia:',zero_crossings[0]*dt-start)
# else:
#     print('Voltage', voltage[i], 'Latencia: None')


