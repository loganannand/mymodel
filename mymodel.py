"""
Code to produce meaningful patterns of spiking neurons, defined by the Izhikevich model.
Author: Logan Annand, lann591@aucklanduni.ac.nz.
"""

from brian2 import *
import matplotlib.pyplot as plt
'''frequencies = []
for i in list(range(10,220, 10)):'''

start_scope()
prefs.codegen.target = 'numpy'
'''frequencies = []
for i in list(range(10, 110, 10)):'''
duration = 250 * ms
defaultclock.dt = 0.001 * ms

I_off = 0 * mV
I_on = 10 * mV
I_app_start = 100 * ms

# ESSneuron_type = ['regular spiking', 'fast spiking', 'regular spiking', 'regular spiking']
'''ESSneuron_type = ['fast spiking']
for i in ESSneuron_type:'''

neuron_type = 'regular spiking'  # Select which neuron type
# Type of neuron
# Columns correspond to a, b, c and d.
abcd_neurons = {'regular spiking': [0.065 / ms, 0.2, -65 * mV, 8 * mV],
                'intrinsically bursting': [0.02 / ms, 0.2, -55 * mV, 4 * mV],
                'chattering': [0.02 / ms, 0.2, -50 * mV, 2 * mV],

                'fast spiking': [0.4 / ms, 0.2, -65 * mV, 2 * mV],
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

'''I_matrix = np.zeros((250, 2))
I_matrix[100:, 0] = 10 * mV
I_app_fn = TimedArray(I_matrix * mV, dt=1*ms)
'''


'''I_array = np.zeros(800)  # 250ms run time
pulses = list(range(50, 751))
I = 10
for i in pulses:
    I_array[i] = I
I_app_fn = I_input = TimedArray(I_array * mV, dt=1 * ms)'''

'''
I = 10
I_array = np.zeros(360)
stim_range = list(range(170, 290))
for i in stim_range:
    I_array[i] = I'''
I_app_fn = TimedArray([I_off, I_on], dt=1 * ms)


G = NeuronGroup(1, IZeqs, threshold='v > v_th', reset=reset_eqs, method='euler')
G.v = c
G.u = 0 * mV

M = StateMonitor(G, variables=['v', 'I_app', 'u'], record=True)
spikemon = SpikeMonitor(G)


#frequencies.append((len(spikemon)/150)*1000)
#### IZHeqs2
IZeqs2 = '''
            dv/dt =   (C1*(v**2) + C2*v + C3 - u + C5*I2_app) / ms: volt
            du/dt = a * (b * v - u) : volt
            I2_app = I2_app_fn(t) : volt
'''
neuron2matrix = np.zeros(250)
I2_app_fn = TimedArray(neuron2matrix * mV, dt = 1*ms)

# Neuron group 2
G2 = NeuronGroup(1, IZeqs2, threshold='v > v_th', reset=reset_eqs, method='euler')
G.v = c
G.u = 0 * mV

Mon2 = StateMonitor(G2, variables=['v'], record=True)
spike2 = SpikeMonitor(G2)


# Synapses
synapse = Synapses(G, G2, 'w:1',
                   on_pre='v_post += w*mV')
synapse.connect(i=0, j=0)
synapse.w = 20
synapse.delay = 2*ms
#synapse.s = 100. * mV


# Monitors
'''M = StateMonitor(G, variables=['v', 'I_app', 'u'], record=True)
spikemon = SpikeMonitor(G)'''
run(duration)

# Plotting monitors
f, axs = plt.subplots(2, 1, figsize=(8, 6), gridspec_kw={'height_ratios': [3, 1]})

axs[0].plot(M.t / ms, M.v[0] / mV, 'k')  # Voltage plot
axs[1].plot(M.t / ms, M[0].I_app / mV, 'k')  # Input current plot
#axs[1].plot(M.t / ms, Mon2[0].I_app / mV, 'r', linestyle='dashed')
# axs[2].plot(M.t / ms, M.u[0] / mV, 'k')  # Reset variable plot
axs[0].plot(M.t / ms, Mon2.v[0] / mV, 'r', linestyle='dashed')  # Voltage plot


axs[0].set_ylabel('Voltage [mV]')
axs[0].set_title('RS', fontsize=18)
axs[1].set_ylabel('Input current [mV] ')  # axs[2].set_ylabel('Reset [u]')
axs[1].set_xlabel('Time [ms]')
'''axs[0].set_xlim([160,300])
axs[1].set_xlim([160,300])'''
#axs[2].set_xlabel('Time [ms]')
axs[0].spines['right'].set_visible(False)
axs[0].spines['top'].set_visible(False)
axs[0].spines['bottom'].set_visible(False)
axs[0].spines['left'].set_visible(False)
axs[1].spines['right'].set_visible(False)
axs[1].spines['top'].set_visible(False)
axs[1].spines['bottom'].set_visible(False)
axs[1].spines['left'].set_visible(False)

# axs[1].spines['bottom'].set_visible(False)
# axs[2].spines['top'].set_visible(False)
# axs[2].spines['right'].set_visible(False)

axs[0].set_xticks([])
axs[1].set_xticks([])
axs[0].set_yticks([])
axs[1].set_yticks([])

plt.show()

print('spike times [ms]', spikemon.t)
print(len(spikemon.t))



