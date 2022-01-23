"""
Code to produce meaningful patterns of spiking neurons, defined by the Izhikevich model.
Author: Logan Annand, lann591@aucklanduni.ac.nz.
"""
# Need to modify the code such that different spiking patterns produce more realistic results, connect neurons first
# Could make four different NeuronGroup(s) each with one of the defined neurons, then can extend the model by
# adding more neurons per layer.
from brian2 import *
import random
import matplotlib.pyplot as plt

start_scope()
prefs.codegen.target = 'numpy'

duration = 250*ms
defaultclock.dt = 0.001*ms

I_off = 0 * mV
I_on = 10 * mV
I_app_start = 20 * ms
#                        MPE,              MPI,             MP5,               THA
#ESSneuron_type = ['regular spiking', 'fast spiking', 'regular spiking', 'regular spiking']
ESSneuron_type = ['regular spiking']
for i in ESSneuron_type:

    neuron_type = 'i'  # Select which neuron type
    # Type of neuron
    # Columns correspond to a, b, c and d.
    abcd_neurons = {'regular spiking': [0.02 / ms, 0.2, -65 * mV, 8 * mV],
                    'intrinsically bursting': [0.02 / ms, 0.2, -55 * mV, 4 * mV],
                    'chattering': [0.02 / ms, 0.2, -50 * mV, 2 * mV],

                    'fast spiking': [0.10 / ms, 0.2, -65 * mV, 2 * mV],
                    'low-threshold spiking': [0.02 / ms, 0.25, -65 * mV, 2 * mV],

                    'thalamo-cortical': [0.02 / ms, 0.25, -65 * mV, 0.05 * mV],
                    'resonator': [0.10 / ms, 0.25, -65 * mV, 8 * mV],
                    }

    a, b, c, d = abcd_neurons[i]
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
    v_0 = c
    u_0 = 0 * mV
    reset_eqs = 'v = c; u = u + d'

    I_app_fn = TimedArray([I_off, I_on], dt=I_app_start)
    N = 1  # Number of neurons
    G = NeuronGroup(N, IZeqs, threshold='v > v_th', reset=reset_eqs, method='euler')
    G.v = v_0
    G.u = u_0

    # Monitors
    M = StateMonitor(G, variables=['v', 'I_app', 'u'], record=True)
    spikemon = SpikeMonitor(G)
    run(duration)

    # Plotting monitors
    pair = np.ones((2, 1))
    fake_spike = [-45, 60]
    f, axs = plt.subplots(2, 1, figsize=(8, 6), gridspec_kw={'height_ratios': [3, 1]})

    axs[0].plot(M.t / ms, M.v[0] / mV, 'k')  # Voltage plot
    axs[1].plot(M.t / ms, M[0].I_app / mV, 'r') # Input current

    axs[0].set_ylabel('voltage [mV]')
    axs[0].set_title(i, fontsize=20)
    axs[0].set_xticklabels([])
    axs[1].set_ylabel('external stim. (mV) ')
    axs[1].set_xlabel('Time (ms)')
    axs[0].spines['right'].set_visible(False)
    axs[0].spines['top'].set_visible(False)
    axs[1].spines['right'].set_visible(False)
    axs[1].spines['top'].set_visible(False)
    plt.savefig(f'izhi_{neuron_type}.png')
    plt.show()



