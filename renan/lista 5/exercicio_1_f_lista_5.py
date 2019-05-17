import scipy as sp
import pylab as plt
from scipy.integrate import odeint, ode
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
start=5
finish=30
def I_inj(t, voltage):  # step up 10 uA/cm^2 every 100ms for 400ms
    return voltage * (t > start) - voltage * (t > finish)
    # return 10*t

# The time to integrate over
dt=0.05
t = sp.arange(0.0, 50.0, dt)

voltage = np.arange(-0.5, -10, -0.5)
plt.figure()
for i in range(len(voltage)):
    plt.plot(t, I_inj(t, voltage[i]))
    plt.xlabel('t (ms)')
    plt.ylabel('$I_{inj}$ ($\\mu{A}/cm^2$)')
plt.legend(voltage)
plt.show()

plt.figure()
for i in range(len(voltage)):
    def dALLdt(X, t):
        V, m, h, n = X

        # calculate membrane potential & activation variables
        dVdt = (I_inj(t,voltage[i]) - I_Na(V, m, h) - I_K(V, n) - I_L(V)) / C_m
        dmdt = alpha_m(V) * (1.0 - m) - beta_m(V) * m
        dhdt = alpha_h(V) * (1.0 - h) - beta_h(V) * h
        dndt = alpha_n(V) * (1.0 - n) - beta_n(V) * n
        return dVdt, dmdt, dhdt, dndt

    #V0 = -65 mV, m(-65) = 0,05,  h(-65) = 0,6, n(-65) = 0,32

    X = odeint(dALLdt, [-65, 0.05, 0.6, 0.32], t)
    V = X[:, 0]

    plt.plot(t, V)

plt.title('Hodgkin-Huxley Neuron')
plt.ylabel('V (mV)')
plt.legend(voltage)
plt.show()


amps=[-1,-6]
for i in range(len(amps)):
    plt.figure()
    def dALLdt(X, t):
        V, m, h, n = X

        # calculate membrane potential & activation variables
        dVdt = (I_inj(t,amps[i]) - I_Na(V, m, h) - I_K(V, n) - I_L(V)) / C_m
        dmdt = alpha_m(V) * (1.0 - m) - beta_m(V) * m
        dhdt = alpha_h(V) * (1.0 - h) - beta_h(V) * h
        dndt = alpha_n(V) * (1.0 - n) - beta_n(V) * n
        return dVdt, dmdt, dhdt, dndt

    #V0 = -65 mV, m(-65) = 0,05,  h(-65) = 0,6, n(-65) = 0,32

    X = odeint(dALLdt, [-65, 0.05, 0.6, 0.32], t)
    V = X[:, 0]
    m = X[:, 1]
    h = X[:, 2]
    n = X[:, 3]
    ina = I_Na(V, m, h)
    ik = I_K(V, n)
    il = I_L(V)

    plt.figure()
    plt.subplot(2, 2, 1)
    plt.plot(t, V, 'k', label='V')
    plt.ylabel('V (mV)')
    plt.legend()
    plt.subplot(2, 2, 2)
    plt.plot(t, n, 'b', label='n')
    plt.legend()

    plt.subplot(2, 2, 3)
    plt.plot(t, m, 'r', label='m')
    plt.ylabel('V (mV)')
    plt.legend()
    plt.subplot(2, 2, 4)
    plt.plot(t, h, 'g', label='h')

    plt.xlabel('Time')
    plt.legend()
    plt.show()