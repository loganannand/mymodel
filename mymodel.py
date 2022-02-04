"""
Code to produce meaningful patterns of spiking neurons, defined by the Izhikevich model.
Author: Logan Annand, lann591@aucklanduni.ac.nz.
"""

from brian2 import *
import matplotlib.pyplot as plt

start_scope()
prefs.codegen.target = 'numpy'
'''frequencies = []
for i in list(range(10, 110, 10)):'''
duration = 250 * ms
defaultclock.dt = 0.001 * ms

I_off = 0 * mV
I_on = 10 * mV
I_app_start = 20 * ms


# ESSneuron_type = ['regular spiking', 'fast spiking', 'regular spiking', 'regular spiking']
'''ESSneuron_type = ['fast spiking']
for i in ESSneuron_type:'''

neuron_type = 'regular spiking'  # Select which neuron type
# Type of neuron
# Columns correspond to a, b, c and d.
abcd_neurons = {'regular spiking': [0.065 / ms, 0.2, -65 * mV, 8 * mV],
                'intrinsically bursting': [0.02 / ms, 0.2, -55 * mV, 4 * mV],
                'chattering': [0.02 / ms, 0.2, -50 * mV, 2 * mV],

                'fast spiking': [0.08 / ms, 0.2, -65 * mV, 2 * mV],
                'low-threshold spiking': [0.02 / ms, 0.25, -65 * mV, 2 * mV],

                'thalamo-cortical': [0.02 / ms, 0.25, -65 * mV, 0.05 * mV],
                'resonator': [0.10 / ms, 0.25, -65 * mV, 8 * mV],
                }
# For RS neuron, changed the following variables: 'a' from 0.02 to 0.065,
# For FS neuron, changed the following variables: 'a' from 0.1 to 0.08
a, b, c, d = abcd_neurons[neuron_type]
v_th = 30 * mV

C1 = 0.04 * 1 / mV
C2 = 5.0
C3 = 140 * mV
C5 = 1.0  # Resistance to injected current

IZeqs = '''
            dv/dt =   (C1*(v**2) + C2*v + C3 - u + C5*I_app) / ms: volt 
            du/dt = a * (b * v - u) : volt
            I_app = I_app_fn(t) : volt
'''

reset_eqs = 'v = c; u = u + d'

#I_app_fn = TimedArray([I_off, I_on], dt=I_app_start) # Replace first argument with array defining spikes. Was '[I_off, I_on]'
I_array = np.zeros(250)
pulses = list(range(20, 230, 15))
I = 20
for i in pulses:
    I_array[i] = I

I_app_fn = I_input = TimedArray(I_array * mV, dt=1 * ms)
N = 1  # Number of neurons
G = NeuronGroup(N, IZeqs, threshold='v > v_th', reset=reset_eqs, method='euler')
G.v = c
G.u = 0 * mV

'''G.v[1] = -65*mV # Initial value of neuron 2?'''

# Freq vs input code
'''inputs = list(range(10, 110, 10)) *mV # Range of input currents from 10mV to 80mV
frequency = []
'''
# Synapses
'''S = Synapses(G, G, 'w : 1', on_pre='v_post += 0.2*mV')
S.connect(i=0, j=1)
S.delay = 5*ms
'''

# Monitors
M = StateMonitor(G, variables=['v', 'I_app', 'u'], record=True)
spikemon = SpikeMonitor(G)
run(duration)
'''frequency.append((len(spikemon)/230)*1000)
print(frequency)'''
#frequencies.append(round((len(spikemon) / 230) * 1000))
# Plotting monitors
f, axs = plt.subplots(3, 1, figsize=(8, 6), gridspec_kw={'height_ratios': [3, 1, 1]})

axs[0].plot(M.t / ms, M.v[0] / mV, 'k')  # Voltage plot
axs[1].plot(M.t / ms, M[0].I_app / mV, 'k')  # Input current plot
axs[2].plot(M.t / ms, M.u[0] / mV, 'k')  # Reset variable plot

# axs[0].plot(M.t / ms, M.v[1] / mV, color='r', linestyle='dashed') # Plot second RS neuron voltage

axs[0].set_ylabel('Voltage [mV]')
axs[0].set_title('IZH MPE neuron', fontsize=20)
axs[1].set_ylabel('Input current [mV] ')  # axs[2].set_ylabel('Reset [u]')
axs[2].set_xlabel('Time [ms]')
axs[0].spines['right'].set_visible(False)
axs[0].spines['top'].set_visible(False)
axs[0].spines['bottom'].set_visible(False)
axs[1].spines['right'].set_visible(False)
axs[1].spines['top'].set_visible(False)
axs[1].spines['bottom'].set_visible(False)
axs[2].spines['top'].set_visible(False)
axs[2].spines['right'].set_visible(False)

axs[0].set_xticks([])
axs[1].set_xticks([])

plt.show()

print('spike times [ms]', spikemon.t)
print(len(spikemon.t))

#print(frequencies)
