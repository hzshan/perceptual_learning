#%%
import matplotlib.pyplot as plt
import numpy as np
import train_utils, glob, torch, pickle, utils, model, copy
from tqdm import trange
import response_utils as r_utils
import theory_utils as t_utils

"""
Created May 28 2020

Solves the self-consistent equations for various parameters, and compare the test_results

Using Haozhe's scaling (W_ij ~ N^-1)

Also using the new OP parameter conventions, u1, v1
"""

%load_ext autoreload
%autoreload 2

# Some global settings
plt.rcParams['figure.dpi'] = 100; torch.set_grad_enabled(False)

parser = train_utils.Args('1D Gabor')
parser.add('nonlinearity', 'relu')
parser.add('loss', 'MSE') # MSE or BCE

# model parameters
parser.add('N', 500); parser.add('Nhid', 500); parser.add('n_layers', 2)

# task parameters
parser.add('sig_w', 0.8); parser.add('sig_s', .2); parser.add('theta', np.pi)

# training parameter
parser.add('eta', 1e-4)
parser.add('n_learn', 2000000)
parser.add('n_train_trials', 500)
parser.add('n_test_trials', 10000) 
parser.add('test_interval', 500)
parser.add('noise_var', 0.01)

args = parser.parse_args()
standard_stim = train_utils.GaborStimuli(args, simple_mode=False)
standard_net = model.Model(args)


#%% Solve the self consistent equations and predict delw for different parameters

k_array = 10.0**np.linspace(-2.5, 2.5, 15)
# k_array = [0.01]
logk = np.log10(k_array)

delw1_list = []
delw2_list = []
OPs = []
active_inds_list = []
a_list = []
full_trajs = []

# parameter-dependent initial guesses for order parameters
u1_guess_log_scale = np.linspace(5.5, 0.1, len(k_array))
v1_guess_log_scale = np.linspace(-5, -1, len(k_array))

for i in trange(len(k_array), position=0):

    k = k_array[i]

    # Load a network and stimuli; get the phi_prime vectors
    net = model.Model(args)
    stimuli = train_utils.GaborStimuli(args, simple_mode=True, verbose=False)

    # if i == 0:
    #     init_guess = [1000, 1e-7]
    # else:
    #     init_guess = _OPs
    init_guess = [10**u1_guess_log_scale[i], 10**v1_guess_log_scale[i]]

    _OPs, delw1, delw2, active_inds, _a, full_OP_traj = t_utils.predict_delw_twolayer(net, stimuli, init_guess, k, update_coef=0.99, Q_epsilon=0, max_iter=2000, convergence_thres=1e-4)

    # _OPs, delw1, delw2, active_inds, __a = t_utils.predict_delw_fsolve_twolayer(net, stimuli, _OPs, k)

    delw1_list.append(delw1); delw2_list.append(delw2); OPs.append(_OPs); a_list.append(_a); active_inds_list.append(active_inds); full_trajs.append(full_OP_traj)


# Plot delw norms as a function of sigma_w
delw1_norms = [np.linalg.norm(delw) for delw in delw1_list]
delw2_norms = [np.linalg.norm(delw) for delw in delw2_list]

plt.figure()
# plt.plot(np.log10(k_array), np.log10(delw1_norms), label='$|| \Delta W_1||$', marker='o')
# plt.plot(np.log10(k_array), np.log10(delw2_norms), label='$|| \Delta W_2||$', marker='o'); plt.title('delw figure')
plt.plot(np.log10(k_array), (delw1_norms), label='$|| \Delta W_1||$')
plt.plot(np.log10(k_array), (delw2_norms), label='$|| \Delta W_2||$'); plt.title('delw figure')
plt.xlabel('log10 k')
plt.legend()
plt.grid()
plt.ylabel('$||\Delta W||$')
plt.savefig('figures/raw/k_delw_two_layer_sigs0P2.svg')
# plt.figure()
# plt.plot(np.array(_OPs)[:, 0]); plt.title('u1')

plt.figure()
plt.plot(np.log10(np.array(OPs)[:, 0]))

plt.figure()
plt.plot(np.log10(np.array(OPs)[:, 1]))



#%% TEMP Plot some full trajectories
trial_ind = 0
plt.figure()
plt.plot(np.array(full_trajs[trial_ind])[:, 0])
plt.figure()
plt.plot(np.array(full_trajs[trial_ind])[:, 1])

#%% Measure linear FI (i.e., transfer with new readouts)
# choose between a regular grid or a finer grid
# probe_angles = np.linspace(0, 2*np.pi, 64, endpoint=False)
probe_angles = np.linspace(1*np.pi, 2*np.pi, 64, endpoint=False)

linear_fi_array = np.zeros((len(k_array), len(probe_angles)))

baseline_linear_fi_array = np.zeros_like(linear_fi_array)

for k_i in trange(len(k_array), position=0):

    _net = model.Model(args)

    baseline_linear_fi_array[k_i, :] = r_utils.get_mean_field_fi(_net, standard_stim)

    t_utils.load_theory_changes(_net, [delw1_list[k_i], delw2_list[k_i]], active_inds_list[k_i])

    for j in range(len(probe_angles)):
        args.theta = probe_angles[j]

        probe_stim = train_utils.GaborStimuli(args, simple_mode=True, verbose=False)
        linear_fi_array[k_i, j] = r_utils.get_mean_field_fi(_net, probe_stim)
args.theta = np.pi

#%%
middle_ind = int(len(k_array) / 2)
plt.figure()
plt.plot(probe_angles, utils.fi_to_error(linear_fi_array)[0], label=f'log10 k={logk[0]:.1f}')
plt.plot(probe_angles, utils.fi_to_error(linear_fi_array)[middle_ind], label=f'log10 k={logk[middle_ind]:.0f}')
plt.plot(probe_angles, utils.fi_to_error(linear_fi_array)[-1], label=f'log10 k={logk[-1]:.1f}')
plt.axhline(utils.fi_to_error(baseline_linear_fi_array[0, 0]), ls='--', color='k')
plt.axhline(stimuli.mld_err, color='r')
plt.title('Error fig')
plt.xlabel('Stimulus')
plt.ylabel('Error fraction')
plt.legend(bbox_to_anchor=(1, 0.3, 0, 0.3))
plt.savefig('figures/raw/transfer_different_k_full_range.svg')
#%% Specificity index figure

del_fi = linear_fi_array - baseline_linear_fi_array
max_diff = linear_fi_array.max(axis=1, keepdims=True) - baseline_linear_fi_array
plt.figure()
plt.plot((del_fi / max_diff)[0])
plt.plot((del_fi / max_diff)[middle_ind])
plt.plot((del_fi / max_diff)[-1])
plt.axhline(0)
# plt.ylim(-0.2, 1)
plt.savefig('figures/raw/2l_specificity_different_k_fine_grid.svg')

# %% Check validity of solution
ind = 0

delw1w2 = []
for ind in range(len(delw1_list)):
    test_net = model.Model(args)
    w_effs = r_utils.get_effective_weights(test_net, standard_stim.x0)[0]
    w1 = w_effs[0] + delw1_list[ind]
    w2 = w_effs[1] + delw2_list[ind]
    plt.figure()
    plt.imshow(w2 @ w1 - w_effs[1] @ w_effs[0])
    delw1w2.append(w2 @ w1 - w_effs[1] @ w_effs[0])

#%% Look at the left and right eigenvectors of changes to W2 W1

_u0s = []

plt.figure()
for i, mat in enumerate(delw1w2):
    _u, _s, _v = np.linalg.svd(mat)
    plt.plot(_u[:, 0], label=k_array[i], color=str(i / (len(k_array) + 1)))
    if k_array[i] == 1:
        plt.plot(_u[:, 0], color='b')
plt.title('left vector')

plt.figure()
for i, mat in enumerate(delw1w2):
    _u, _s, _v = np.linalg.svd(mat)
    plt.plot(_v[0, :], label=k_array[i], color=str(i / (len(k_array) + 1)))
    if k_array[i] == 1:
        plt.plot(_v[0, :], color='b')
plt.title('right vector')
# print(np.linalg.norm(utils.recover_matrix(dataset[7]['delW'][0][-1])))
#%% 
delw1_fig = plt.figure(figsize=(7, 2))
delw1_fig.add_subplot(131)
plt.imshow(delw1_list[0] * args.N); plt.colorbar(orientation='horizontal'); plt.title(f'k={k_array[0]}')
delw1_fig.add_subplot(132)
plt.imshow(delw1_list[4] * args.N); plt.colorbar(orientation='horizontal'); plt.title(f'k={k_array[4]}')
delw1_fig.add_subplot(133)
plt.imshow(delw1_list[-1] * args.N); plt.colorbar(orientation='horizontal'); plt.title(f'k={k_array[-1]}')
plt.suptitle('$\Delta W1 * N$')
plt.tight_layout(); plt.subplots_adjust(top=0.8)

delw2_fig = plt.figure(figsize=(7, 2))
delw2_fig.add_subplot(131)
plt.imshow(delw2_list[0] * args.N); plt.colorbar(orientation='horizontal'); plt.title(f'k={k_array[0]}')
delw2_fig.add_subplot(132)
plt.imshow(delw2_list[4] * args.N); plt.colorbar(orientation='horizontal'); plt.title(f'k={k_array[4]}')
delw2_fig.add_subplot(133)
plt.imshow(delw2_list[-1] * args.N); plt.colorbar(orientation='horizontal'); plt.title(f'k={k_array[-1]}')
plt.suptitle('$\Delta W2 * N$')
plt.tight_layout(); plt.subplots_adjust(top=0.7)

#%% look at compositions of the weights
w10, w20 = r_utils.get_effective_weights(standard_net, standard_stim.x0)[0]
delw1_x1_proj = [np.linalg.norm(mat @ standard_stim.x1_normed.t().numpy()) / np.linalg.norm(mat) for mat in delw1_list]

x11 = w10 @ standard_stim.x1_normed.t().numpy()
x11 /= np.linalg.norm(x11)
delw2_x1_proj = [np.linalg.norm(mat @ x11) / np.linalg.norm(mat) for mat in delw2_list]

plt.figure()
plt.plot(logk, delw1_x1_proj)
plt.plot(logk, delw2_x1_proj)
#%% Check stability of solutions
for i in range(len(k_array)):
    print(k_array[i] - np.linalg.norm(a_list[0])**2 * OPs[i][1])

#%% Get tuning properties (REQUIRED for most of the cells below)
# Create a list of dictionaries
response_properties = [{'pref':[], 'max':[], 'mean':[], 'mean0':[],'mean_over_stim':[], 'max0':[], 'slope':[], 'slope0':[], 'snr':[], 'snr0':[], 'noise':[], 'noise0':[], 'tuning_params':[], 'tuning_params0':[]} for i in range(standard_net.n_layers)]

for i in trange(len(k_array)):
    _net = model.Model(args)
    _net2 = model.Model(args)
    t_utils.load_theory_changes(_net2, [delw1_list[i], delw2_list[i]], active_inds_list[i])

    r_utils.compute_response_properties(_net, _net2, standard_stim, response_properties)


#%% Histogram of preferred orientations
r_utils.preferred_orientation_histogram_figure(response_properties[0]); plt.title('First layer')

r_utils.preferred_orientation_histogram_figure(response_properties[1]); plt.title('Second layer')

#%% KL divergence figure of how much the distribution of preferred orientations changed
from scipy.stats import entropy, wasserstein_distance

nbins = 10
bl_dist = np.ones(nbins)
kl_div_l1_list = []
kl_div_l2_list = []

for i in range(len(k_array)):
    hist, bin_edges = np.histogram(response_properties[0]['pref'][i], bins=np.linspace(-0.1, response_properties[0]['pref'][i].shape[0]-.5, nbins+1))
    kl_div_l1_list.append(entropy(hist, bl_dist))
    hist, bin_edges = np.histogram(response_properties[1]['pref'][i], bins=np.linspace(-0.1, response_properties[0]['pref'][i].shape[0]-.5, nbins+1))
    kl_div_l2_list.append(entropy(hist, bl_dist))

plt.figure()
plt.plot(k_array, kl_div_l1_list, label='First layer')
plt.plot(k_array, kl_div_l2_list, label='Second layer')
plt.title('$D_{KL}$ between distributions of POs' + f' #bins:{nbins}'); plt.legend()
plt.xlabel('$\sigma_w$')

#%% Wasserstein distance figure of how much the distribution of preferred orientations changed
from scipy.stats import entropy, wasserstein_distance

bl_dist = np.linspace(0, args.N, args.N-1, endpoint=False)
wasserstein_l1_list = []
wasserstein_l2_list = []

for i in range(len(k_array)):
    wasserstein_l1_list.append(wasserstein_distance(response_properties[0]['pref'][i], bl_dist))
    wasserstein_l2_list.append(wasserstein_distance(response_properties[1]['pref'][i], bl_dist))

plt.figure()
plt.plot(k_array, wasserstein_l1_list, label='First layer')
plt.plot(k_array, wasserstein_l2_list, label='Second layer')
plt.title('Wasserstein distance between distributions of POs'); plt.legend()
plt.xlabel('$\sigma_w$')
#%% Changes to absolute slope

r_utils.slope_at_trained_stimulus_figure(response_properties[0]); plt.title('First layer')
plt.ylabel('k'); plt.yticks([2, 0], [np.min(k_array), np.max(k_array)])
r_utils.slope_at_trained_stimulus_figure(response_properties[1]); plt.title('Second layer')
plt.ylabel('k'); plt.yticks([2, 0], [np.min(k_array), np.max(k_array)])

#%% Changes to relative slope

r_utils.relative_slope_at_trained_stimulus_figure(response_properties[0]); plt.title('First layer')

r_utils.relative_slope_at_trained_stimulus_figure(response_properties[1]); plt.title('Second layer')

#%% Changes to max firing

r_utils.max_firing_fig(response_properties[0]); plt.title('First layer')

r_utils.max_firing_fig(response_properties[1]); plt.title('Second layer')

#%% Changes to mean firing

r_utils.mean_firing_fig(response_properties[0]); plt.title('First layer')

r_utils.mean_firing_fig(response_properties[1]); plt.title('Second layer')

#%% Changes to single neuron SNR

r_utils.single_neuron_snr_figure(response_properties, 0); plt.title('First layer')
plt.ylabel('log10 k'); plt.yticks([2, 1, 0], [np.min(logk), 0, np.max(logk)])

r_utils.single_neuron_snr_figure(response_properties, 1); plt.title('Second layer')
plt.ylabel('log10 k'); plt.yticks([2, 1, 0], [np.min(logk), 0, np.max(logk)])

#%%
plt.figure()
plt.plot(logk, np.log10(OPs)[:, 0]); plt.title('u1')
plt.figure()
plt.plot(logk, np.log10(np.array(OPs)[:, 1])); plt.title('v1')