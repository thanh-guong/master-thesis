import matplotlib.pyplot as plt
import numpy as np

'''
def f(t):
    return np.exp(-t) * np.cos(2*np.pi*t)

t1 = np.arange(0.0, 5.0, 0.1)
t2 = np.arange(0.0, 5.0, 0.02)

plt.figure()
plt.subplot(211)
plt.plot(t1, f(t1), 'bo', t2, f(t2), 'k')

plt.subplot(212)
plt.plot(t2, np.cos(2*np.pi*t2), 'r--')
plt.show()
'''

plt.figure()

time_scale = np.arange(0, 20, 0.1) # range [0, 5] stepping by 0.1
modulating_signal = 2
carrier_signal = np.sin(time_scale)
modulated_signal = carrier_signal * modulating_signal

plt.plot(time_scale, modulating_signal)
plt.plot(time_scale, carrier_signal)
plt.plot(time_scale, modulated_signal)

plt.show()