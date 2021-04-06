#%%
import matplotlib.pyplot as plt
import numpy as np
import train_utils, torch, pickle, utils, model, copy
from tqdm import trange, tqdm
import response_utils as r_utils
import theory_utils as t_utils

"""
Try out some manipulations where the readout is fixed.
"""

%load_ext autoreload
%autoreload 2

torch.set_grad_enabled(False)

parser = train_utils.Args('1D Gabor')
parser.add('nonlinearity', 'relu')
parser.add('loss', 'MSE') # MSE or BCE

# model parameters
parser.add('N', 1000); parser.add('Nhid', 1000); parser.add('n_layers', 3)

# task parameters
parser.add('sig_w', 0.1); parser.add('sig_s', 1.2); parser.add('theta', np.pi)

# training parameter
parser.add('eta', 1e-4)
parser.add('n_learn', 2000000)
parser.add('n_train_trials', 500)
parser.add('n_test_trials', 10000)
parser.add('test_interval', 500)
parser.add('noise_var', .01)

args = parser.parse_args()

# The sig_w values to test
# sig_w_array = np.linspace(0.1, 0.6, 30)
standard_stim = train_utils.GaborStimuli(args, simple_mode=True)

#%% Load three layer theory results

# file_name = 'Saved Results/THEORY_RESULTS_3_layers_sigs1P2'
file_name = 'Saved Results/theory_3L_sigs1P2'

theory_dict = pickle.load(open(file_name, 'rb'))
OPs = theory_dict['OP']
delw1_list = theory_dict['delw1']; delw2_list = theory_dict['delw2']
delw3_list = theory_dict['delw3']
active_inds_list = theory_dict['active_inds']
a_list = theory_dict['a']
sig_w_array = theory_dict['sig_w']

#%% Transfer while fixing the readout
"""
For each test angle theta, use the readout adapted to the trained weights and theta. Then force the active neurons to be the same as those active for theta before training. Use the aforementioned readout to test accuracy.
"""

sig_w_ind = 20
probe_args = copy.copy(args); probe_args.sig_w = sig_w_array[sig_w_ind]

# create a pair of networks
bl_net = model.Model(probe_args)
trained_net = model.Model(probe_args)

t_utils.load_theory_changes(trained_net, [delw1_list[sig_w_ind], delw2_list[sig_w_ind], delw3_list[sig_w_ind]], active_inds_list[sig_w_ind])

probe_angles = np.linspace(np.pi, 2*np.pi, 64, endpoint=False)

bl_info = np.zeros_like(probe_angles)
trained_info = np.zeros_like(bl_info)
trained_manipulated_info = np.zeros_like(bl_info)

for i in trange(len(probe_angles)):
    probe_args.theta = probe_angles[i]
    probe_stim = train_utils.GaborStimuli(probe_args, verbose=False, simple_mode=True)
    bl_active_inds = r_utils.get_effective_weights(bl_net, probe_stim.x0)[1]

    # first, get the un-manipulated readout
    w_effs = r_utils.get_effective_weights(trained_net, probe_stim.x0, full_mat=True)[0]
    forward_mat = w_effs[2] @ w_effs[1] @ w_effs[0]
    readout = np.linalg.inv(forward_mat @ forward_mat.T * probe_args.noise_var + 1e-9 * np.eye(forward_mat.shape[0])) @ forward_mat @ probe_stim.x1.t().numpy()

    # manipulated forward mat
    manipulated_Ws = []
    for j in range(3):
        mask = np.zeros_like(trained_net.Ws[j])
        mask[bl_active_inds[j+1]] = 1
        manipulated_Ws.append(trained_net.Ws[j] * mask)
    
    manipulated_forward_mat = manipulated_Ws[2] @ manipulated_Ws[1] @ manipulated_Ws[0]
    trained_manipulated_signal = float(readout.T @ manipulated_forward_mat @ probe_stim.x1.t().numpy()) **2
    trained_manipulated_noise = float(readout.T @ manipulated_forward_mat @ manipulated_forward_mat.T @ readout) * probe_args.noise_var
    trained_manipulated_info[i] = trained_manipulated_signal / (trained_manipulated_noise + 1e-5)
    trained_info[i] = float(readout.T @ forward_mat @ probe_stim.x1.t().numpy())

#%%
plt.figure()
plt.plot(probe_angles, trained_info, label='without opto')
plt.plot(probe_angles, trained_manipulated_info, label='with opto')
plt.ylabel('J / J_input')
plt.xlabel('Stimulus')
plt.legend()
plt.title(f'sigma_w={sig_w_array[sig_w_ind]}')
#%% Transfer while fixing the active neurons to those active for TO

sig_w_ind = 23
probe_args = copy.copy(args); probe_args.sig_w = sig_w_array[sig_w_ind]

# create a pair of networks
trained_net = model.Model(probe_args)
bl_net = model.Model(probe_args)

# load changes. Comment this line out to use untrained weights
t_utils.load_theory_changes(trained_net, [delw1_list[sig_w_ind], delw2_list[sig_w_ind], delw3_list[sig_w_ind]], active_inds_list[sig_w_ind])


probe_angles = np.linspace(np.pi, 1.5*np.pi, 64, endpoint=False)

bl_info = np.zeros_like(probe_angles)
trained_info = np.zeros_like(bl_info)
trained_manipulated_info = np.zeros_like(bl_info)

probe_args.theta = np.pi

# indices of active neurons for the trained stimulus
probe_stim = train_utils.GaborStimuli(probe_args, verbose=False, simple_mode=True)
to_active_inds = r_utils.get_effective_weights(trained_net, probe_stim.x0)[1]
manipulated_Ws = []

for j in range(3):
    mask = np.zeros_like(trained_net.Ws[j])
    mask[to_active_inds[j+1]] = 1
    manipulated_Ws.append(trained_net.Ws[j] * mask)
manipulated_forward_mat = manipulated_Ws[2] @ manipulated_Ws[1] @ manipulated_Ws[0]

for i in trange(len(probe_angles)):
    probe_args.theta = probe_angles[i]
    probe_stim = train_utils.GaborStimuli(probe_args, verbose=False, simple_mode=True)
    bl_info[i] = r_utils.get_mean_field_fi(bl_net, probe_stim, epsilon=1e-6)

    # first, get the un-manipulated readout
    w_effs = r_utils.get_effective_weights(trained_net, probe_stim.x0, full_mat=True)[0]
    forward_mat = w_effs[2] @ w_effs[1] @ w_effs[0]
    readout = np.linalg.inv(forward_mat @ forward_mat.T * probe_args.noise_var + probe_args.noise_var * 1e-6 * np.eye(forward_mat.shape[0])) @ forward_mat @ probe_stim.x1.t().numpy()

    # manipulated forward mat
    trained_manipulated_signal = float(readout.T @ manipulated_forward_mat @ probe_stim.x1.t().numpy())**2
    trained_manipulated_noise = float(readout.T @ manipulated_forward_mat @ manipulated_forward_mat.T @ readout) * probe_args.noise_var
    trained_manipulated_info[i] = trained_manipulated_signal / (trained_manipulated_noise + 1e-9)
    trained_info[i] = float(readout.T @ forward_mat @ probe_stim.x1.t().numpy())
    

#%% 

def convert_to_specificity(fi_array, baseline_fi):
    return (fi_array - baseline_fi) / (np.max(fi_array - baseline_fi))

trained_manipulated_info[trained_manipulated_info < 0] = 0

plt.figure()
plt.plot(probe_angles, bl_info)
plt.plot(probe_angles, trained_info)
plt.plot(probe_angles, trained_manipulated_info)

plt.figure()
plt.plot(probe_angles, convert_to_specificity(trained_info, bl_info))
plt.plot(probe_angles, convert_to_specificity(trained_manipulated_info, bl_info))
plt.axhline(0)
plt.savefig('figures/raw/opto_transfer_across_theta.svg')

#%% Transfer across sigs, with optogenetics

sig_w_ind = 23
probe_args = copy.copy(args); probe_args.sig_w = sig_w_array[sig_w_ind]

# create a pair of networks
bl_net = model.Model(probe_args)
trained_net = model.Model(probe_args)

t_utils.load_theory_changes(trained_net, [delw1_list[sig_w_ind], delw2_list[sig_w_ind], delw3_list[sig_w_ind]], active_inds_list[sig_w_ind])

sig_s_array = np.linspace(0.1, 1.1, 30)

bl_info = np.zeros_like(sig_s_array)
trained_info = np.zeros_like(bl_info)
trained_manipulated_info = np.zeros_like(bl_info)

for i in trange(len(sig_s_array)):
    probe_args.sig_s = sig_s_array[i]
    probe_stim = train_utils.GaborStimuli(probe_args, verbose=False, simple_mode=True)
    bl_active_inds = r_utils.get_effective_weights(bl_net, probe_stim.x0)[1]
    bl_info[i] = r_utils.get_mean_field_fi(bl_net, probe_stim)

    # first, get the un-manipulated readout
    w_effs = r_utils.get_effective_weights(trained_net, probe_stim.x0, full_mat=True)[0]
    forward_mat = w_effs[2] @ w_effs[1] @ w_effs[0]
    readout = np.linalg.inv(forward_mat @ forward_mat.T * probe_args.noise_var + 1e-9 * np.eye(forward_mat.shape[0])) @ forward_mat @ probe_stim.x1.t().numpy()

    # manipulated forward mat
    manipulated_Ws = []
    for j in range(3):
        mask = np.zeros_like(trained_net.Ws[j])
        mask[bl_active_inds[j+1]] = 1
        manipulated_Ws.append(trained_net.Ws[j] * mask)
    
    manipulated_forward_mat = manipulated_Ws[2] @ manipulated_Ws[1] @ manipulated_Ws[0]
    trained_manipulated_signal = float(readout.T @ manipulated_forward_mat @ probe_stim.x1.t().numpy()) **2
    trained_manipulated_noise = float(readout.T @ manipulated_forward_mat @ manipulated_forward_mat.T @ readout) * probe_args.noise_var
    trained_manipulated_info[i] = trained_manipulated_signal / (trained_manipulated_noise + 1e-5)
    trained_info[i] = float(readout.T @ forward_mat @ probe_stim.x1.t().numpy())

#%%
plt.figure()
plt.plot(sig_s_array, trained_info, label='without opto')
plt.plot(sig_s_array, trained_manipulated_info, label='with opto')
plt.plot(sig_s_array, bl_info, label='baseline')
plt.legend(); plt.xlabel('')
plt.title(f'sigw={sig_w_array[sig_w_ind]}')
plt.axvline(0.2)
plt.savefig('figures/raw/three_layer_transfer_acorss_sigs.svg')

