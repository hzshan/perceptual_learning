#%%
import matplotlib.pyplot as plt
import numpy as np
import train_utils, torch, pickle, utils, model, copy
from tqdm import trange, tqdm
import response_utils as r_utils
import theory_utils as t_utils

%load_ext autoreload
%autoreload 2

'''Created Sept 17, 2020'''

parser = train_utils.Args('1D Gabor')
parser.add('nonlinearity', 'relu')
parser.add('loss', 'MSE') # MSE or BCE

# model parameters
parser.add('N', 1000); parser.add('Nhid', 1000); parser.add('n_layers', 3)

# task parameters
parser.add('sig_w', 0.1); parser.add('sig_s', 0.2); parser.add('theta', np.pi)

parser.add('n_learn', 2000000)
parser.add('n_train_trials', 500)
parser.add('n_test_trials', 10000) 
parser.add('noise_var', .01)

args = parser.parse_args()

probe_angles = np.linspace(np.pi, 2*np.pi, 64)

file_name = 'Saved Results/theory_3L_sigs0P2'
# MAKE sure the default parameters are updated accordingly

theory_dict = pickle.load(open(file_name, 'rb'))
delw1_list = theory_dict['delw1']; delw2_list = theory_dict['delw2']
delw3_list = theory_dict['delw3']
active_inds_list = theory_dict['active_inds']
a_list = theory_dict['a']
sig_w_array = theory_dict['sig_w']

sig_w_ind = 23

probe_args = copy.copy(args)
probe_args.sig_w = sig_w_array[sig_w_ind]

# create a network
standard_stim = train_utils.GaborStimuli(args, simple_mode=True, verbose=False)
bl_net = model.Model(probe_args)
trained_net = model.Model(probe_args)
t_utils.load_theory_changes(trained_net, [delw1_list[sig_w_ind], delw2_list[sig_w_ind], delw3_list[sig_w_ind]], active_inds_list[sig_w_ind])

def transfer_index_curve(info_curve, baseline_info_curve):
    return (info_curve - baseline_info_curve) / np.max(info_curve - baseline_info_curve)

#%% COMPARE TRANSFER ACROSS LAYERS

linear_fi_array = np.zeros((3, len(probe_angles)))
baseline_linear_fi_array = np.zeros_like(linear_fi_array)

for i in [1, 2, 3]:

    baseline_linear_fi_array[i-1, :] = r_utils.get_mean_field_fi(bl_net, standard_stim, to_layer=i)

    for j in range(len(probe_angles)):
        probe_args.theta = probe_angles[j]
        probe_stim = train_utils.GaborStimuli(probe_args, simple_mode=True, verbose=False)
        linear_fi_array[i-1, j] = r_utils.get_mean_field_fi(trained_net, probe_stim, to_layer=i)

# calculate specificity curves
plt.figure()
for i in range(3):
    plt.plot(transfer_index_curve(linear_fi_array[i], baseline_linear_fi_array[i]))

plt.axhline(0)


# plt.savefig('figures/raw/3layer_specificity_each_layer_0P5pi_1P5pi.svg')
# plt.savefig('figures/raw/3layer_specificity_each_layer_0pi_2pi.svg')
#%% COMPARE NUMBER OF ACTIVE NEURONS ACROSS LAYERS

n_active = np.zeros((3, len(probe_angles)))
bl_n_active = np.zeros_like(n_active)

for i in [1, 2, 3]:

    bl_n_active[i-1, :] = r_utils.get_effective_weights(bl_net, standard_stim.x0)[0][i-1].shape[0]

    for j in range(len(probe_angles)):
        probe_args.theta = probe_angles[j]
        probe_stim = train_utils.GaborStimuli(probe_args, simple_mode=True, verbose=False)
        n_active[i-1, j] = r_utils.get_effective_weights(trained_net, probe_stim.x0)[0][i-1].shape[0]

plt.figure()
for i in range(3):
    plt.plot(probe_angles, bl_n_active[i] / args.N, color='C' + str(i), ls='--')
    plt.plot(probe_angles, n_active[i] / args.N, color='C' + str(i))
plt.xlabel('Stimulus')
plt.ylabel('fraction of active neurons')
plt.title(f'sigma_w={probe_args.sig_w}')
# plt.savefig('figures/raw/active_neurons_num_3l.svg')

#%% IS THE NUMBER OF ACTIVE NEURONS CAUSE FOR FAR TRANSFER?
# Scheme: Use the learnt matrices, but keep the same active neurons as pre-PL.

fixed_active_fi_arr = np.zeros((3, len(probe_angles)))

for to_layer in [1, 2, 3]:

    for j in range(len(probe_angles)):
        probe_args.theta = probe_angles[j]
        probe_stim = train_utils.GaborStimuli(probe_args, simple_mode=True, verbose=False)
        active_inds = r_utils.get_effective_weights(bl_net, probe_stim.x0)[1]
        W0 = trained_net.Ws[0][active_inds[1]]
        W1 = trained_net.Ws[1][active_inds[2]][:, active_inds[1]]
        W2 = trained_net.Ws[2][active_inds[3]][:, active_inds[2]] 

        effective_Ws = [W0, W1, W2]
        forward_mat = effective_Ws[0]
        for i in np.arange(1, to_layer):
            forward_mat = effective_Ws[i] @ forward_mat

        slope = forward_mat @ probe_stim.x1.cpu().t().numpy()
        covar = forward_mat @ forward_mat.T * probe_stim.noise_var
        _u, _s, _v = np.linalg.svd(covar)

        fixed_active_fi_arr[to_layer-1, j] = float(slope.T @ np.linalg.inv(covar + _s[0] * 1e-6 * np.eye(covar.shape[0])) @ slope)


layer_ind = 2
plt.figure()
plt.plot(transfer_index_curve(linear_fi_array[layer_ind], baseline_linear_fi_array[layer_ind]), label='wild type')
plt.plot(transfer_index_curve(fixed_active_fi_arr[layer_ind], baseline_linear_fi_array[layer_ind]), label='fixed active')
plt.legend(bbox_to_anchor=(1.01, 1))
plt.axhline(0)

np.sum(np.abs(transfer_index_curve(fixed_active_fi_arr[layer_ind], baseline_linear_fi_array[layer_ind]))) / np.sum(np.abs(transfer_index_curve(linear_fi_array[layer_ind], baseline_linear_fi_array[layer_ind])))
# plt.savefig('figures/raw/fixed_active_inds_tranfer_3l_first_layer.svg')

# %% Compare active inds for some other angle before and after training

theta_te = np.pi * 1.10
probe_args.theta = theta_te
stim_te = train_utils.GaborStimuli(probe_args, simple_mode=True, verbose=False)

bl_act_inds = r_utils.get_active_inds(bl_net, stim_te.x0)
tr_act_inds = r_utils.get_active_inds(trained_net, stim_te.x0) 

plt.figure()
_axis = np.zeros(1000)
_axis[bl_act_inds[-1]] = 1
plt.plot(_axis)

_axis = np.zeros(1000)
_axis[tr_act_inds[-1]] = 1
plt.plot(_axis)
