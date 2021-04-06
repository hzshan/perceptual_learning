#%%
import matplotlib.pyplot as plt
import numpy as np
import train_utils, glob, torch, pickle, utils, model, copy
from tqdm import trange
import response_utils as r_utils
import theory_utils as t_utils

"""
Solves constrained-optimization theory for two layer networks.
Last updated: Sept 4.
"""

%load_ext autoreload
%autoreload 2

# Some global settings
plt.rcParams['figure.dpi'] = 100
torch.set_grad_enabled(False)

parser = train_utils.Args('1D Gabor')
parser.add('nonlinearity', 'relu')
parser.add('loss', 'MSE') # MSE or BCE

# model parameters
parser.add('N', 1000); parser.add('Nhid', 1000); parser.add('n_layers', 2)

# task parameters
parser.add('sig_w', 0.5); parser.add('sig_s', 0.6); parser.add('theta', np.pi)
parser.add('noise_var', .01)

# Pro Forma parameters
parser.add('eta', 1e-4); parser.add('n_learn', 2000000); parser.add('n_train_trials', 500)
parser.add('n_test_trials', 10000); parser.add('test_interval', 500)

args = parser.parse_args()

standard_stim = train_utils.GaborStimuli(args, simple_mode=False)
standard_net = model.Model(args)

k = 1  # relative strength of the L2 constraints. Default is 1. 

#%% Solve the self consistent equations and predict delw for different parameters
'''This cell solves the scalar self-consistent equations for the theory. It iterates the self-consistent equations from some initial conditions. The important numerical parameters here are the initial guesses, and the step size. The initial guesses are set here. The step size is set in the theory_utils.py file. By default, after each iteration, the new value is 0.9 * old value + 0.1 * new value.

load_saved_results: Boolean, whether or not to load saved solutions
save_results: Boolean, whether or not save solutions

'''
sig_w_array = np.linspace(0.1, 1.0, 30) # use for sig_s=1.2
# sig_w_array = np.linspace(0.5, 1.0, 30) # use for sig_s=0.2
load_saved_results = False
save_results = True
file_path = 'Saved Results/theory_2L_sigs0P6'
# file_path = 'Saved Results/THEORY_RESULTS_2_layers'

if load_saved_results:
    theory_dict = pickle.load(open(file_path, 'rb'))
    delw1_list = theory_dict['delw1']; delw2_list = theory_dict['delw2']; a_list = theory_dict['a']
    sig_w_array = theory_dict['sig_w']; active_inds_list = theory_dict['active_inds']
    OPs = theory_dict['OP']
else:
    delw1_list = []; delw2_list = []; OPs = []; active_inds_list = []; a_list = []
    trajectories = []
    temp_args = copy.copy(args)

    for i in trange(len(sig_w_array), position=0):

        temp_args.sig_w = sig_w_array[i]

        # Load a network and stimuli
        net = model.Model(temp_args)
        stimuli = train_utils.GaborStimuli(temp_args, simple_mode=True, verbose=False)

        # Uses solutions for the previous parameter as the initial guess
        if i == 0:
            init_guess = [140, 0.00001] # for sigs=0.2
        else:
            init_guess = OPs[-1]

        _OPs, delw1, delw2, active_inds, _a, trajs = t_utils.predict_delw_twolayer(net, stimuli, k=k, init_guess=init_guess, update_coef=0.99, Q_epsilon=0, max_iter=100, convergence_thres=1e-4)

        _OPs, delw1, delw2, active_inds, _a = t_utils.predict_delw_fsolve_twolayer(net, stimuli, _OPs, k=1)

        delw1_list.append(delw1); delw2_list.append(delw2)

        OPs.append(_OPs); a_list.append(_a); active_inds_list.append(active_inds)
        trajectories.append(trajs)

    if save_results:
        pickle.dump({'delw1':delw1_list, 'delw2':delw2_list, 'OP':OPs, 'a':a_list, 'sig_w':sig_w_array, 'active_inds':active_inds_list}, open(file_path, 'wb'))


#%% TEMP check some trajectories
plt.figure()
plt.plot(np.array(trajectories[20])[:, 1])
#%% Compute and plot delw norms as a function of sigma_w
''' Compute the norm of delW matrices from simulations and theory. '''

data_sigw = []
data_delw1_norm = []
data_delw2_norm = []

# header = 'Grad Descent'

# for _data in dataset:
#     data_sigw.append(_data['args'].sig_w)
#     data_delw1_norm.append(np.linalg.norm(utils.recover_matrix(_data['delW'][0][-1])))
#     data_delw2_norm.append(np.linalg.norm(utils.recover_matrix(_data['delW'][1][-1])))

# Compute delW norms for theoretical predictions
delw1_norms = [np.linalg.norm(delw) for delw in delw1_list]
delw2_norms = [np.linalg.norm(delw) for delw in delw2_list]

normalized_delw1_norms = [np.linalg.norm(delw) / np.sqrt(len(active_inds[1]) / len(active_inds[0])) for delw, active_inds in zip(delw1_list, active_inds_list)]
normalized_delw2_norms = [np.linalg.norm(delw) / np.sqrt(len(active_inds[2]) / len(active_inds[1])) for delw, active_inds in zip(delw2_list, active_inds_list)]


plt.figure()
# plt.scatter(np.sort(data_sigw), np.array(data_delw1_norm)[np.argsort(data_sigw)], marker='o', label=header + '$ || \Delta W_1 ||$')
# plt.scatter(np.sort(data_sigw), np.array(data_delw2_norm)[np.argsort(data_sigw)], marker='o', label=header + '$|| \Delta W_2 ||$')

plt.xlabel('$\sigma_w$')

plt.plot(sig_w_array, delw1_norms, ls='--', label='Theory $|| \Delta W_1 || $')
plt.plot(sig_w_array, delw2_norms, ls='--', label='Theory $|| \Delta W_2 || $')

plt.legend(bbox_to_anchor=(1.01, 1))
# plt.savefig('figures/two_layer_theory_with_data.svg')

plt.figure()
plt.plot(sig_w_array, normalized_delw1_norms, label='$||\Delta w_1|| / ||W_{1eff}||$')
plt.plot(sig_w_array, normalized_delw2_norms, label='$||\Delta w_2|| / ||W_{2eff}||$')
plt.xlabel('$\sigma_w$')
plt.legend(); plt.grid()

"""Plot the order parameters as function of sigma_w"""
order_param_fig = plt.figure()
order_param_fig.add_subplot(211)
plt.plot(sig_w_array, np.array(OPs)[:, 0], label='u1'); plt.legend()

order_param_fig.add_subplot(212)
plt.plot(sig_w_array, np.array(OPs)[:, 1], label='v1')
plt.xlabel('sig_w'); plt.legend()

#%%
plt.figure()
plt.plot([np.linalg.norm(_a) for _a in])

#%% Get tuning properties (REQUIRED for most of the cells below)
'''This cell calculates various response property parameters from the theory and store them for plotting in subsequent cells.'''

# Create a list of dictionaries
response_properties = [{'pref':[], 'max':[], 'mean':[], 'mean0':[],'mean_over_stim':[], 'max0':[], 'slope':[], 'slope0':[], 'snr':[], 'snr0':[], 'noise':[], 'noise0':[], 'tuning_params':[], 'tuning_params0':[], 'covar':[], 'covar0':[]} for i in range(standard_net.n_layers)]

for i in trange(len(sig_w_array)):
    args.sig_w = sig_w_array[i]
    _net = model.Model(args)
    _net2 = model.Model(args)
    t_utils.load_theory_changes(_net2, [delw1_list[i], delw2_list[i]], active_inds_list[i])
    r_utils.compute_response_properties(_net, _net2, standard_stim, response_properties)

#%% Histogram of preferred orientations
r_utils.preferred_orientation_histogram_figure(response_properties[0]); plt.title('First layer')

r_utils.preferred_orientation_histogram_figure(response_properties[1]); plt.title('Second layer')

#%% TEMP
theory_ind = 5
plt.figure()
plt.plot(response_properties[1]['pref'][theory_ind], response_properties[1]['slope'][theory_ind])
plt.plot(response_properties[1]['slope0'][theory_ind])
#%% Wasserstein distance figure of how much the distribution of preferred orientations changed
from scipy.stats import entropy, wasserstein_distance

bl_dist = np.linspace(0, args.N, args.N-1, endpoint=False)
wasserstein_l1_list = []
wasserstein_l2_list = []

for i in range(len(sig_w_array)):
    wasserstein_l1_list.append(wasserstein_distance(response_properties[0]['pref'][i], bl_dist))
    wasserstein_l2_list.append(wasserstein_distance(response_properties[1]['pref'][i], bl_dist))

plt.figure()
plt.plot(sig_w_array, wasserstein_l1_list, label='First layer')
plt.plot(sig_w_array, wasserstein_l2_list, label='Second layer')
plt.title('Wasserstein distance between distributions of POs'); plt.legend()
plt.xlabel('$\sigma_w$')
#%% Changes to absolute slope
r_utils.slope_at_trained_stimulus_figure(response_properties[0]); plt.title('First layer')

r_utils.slope_at_trained_stimulus_figure(response_properties[1]); plt.title('Second layer')
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

r_utils.single_neuron_snr_figure(response_properties, 1); plt.title('Second layer')

#%% single neuron noise

r_utils.single_neuron_noise_figure(response_properties, 0); plt.title('First layer')
r_utils.single_neuron_noise_figure(response_properties, 1); plt.title('Second layer')

#%% Measure linear FI (i.e., transfer with new readouts) from theory

probe_angles = np.linspace(0, 2*np.pi, 32, endpoint=False)
linear_fi_array = np.zeros((len(sig_w_array), len(probe_angles)))
baseline_linear_fi_array = np.zeros_like(linear_fi_array)
probe_args = copy.copy(args)

for sig_w_ind in trange(len(sig_w_array), position=0):

    probe_args.sig_w = sig_w_array[sig_w_ind]
    _net = model.Model(probe_args)

    baseline_linear_fi_array[sig_w_ind, :] = r_utils.get_mean_field_fi(_net, standard_stim)

    t_utils.load_theory_changes(_net, [delw1_list[sig_w_ind], delw2_list[sig_w_ind]], active_inds_list[sig_w_ind])

    for j in range(len(probe_angles)):
        probe_args.theta = probe_angles[j]

        probe_stim = train_utils.GaborStimuli(probe_args, simple_mode=True, verbose=False)
        linear_fi_array[sig_w_ind, j] = r_utils.get_mean_field_fi(_net, probe_stim, epsilon=1e-9)

#%% Figure making of the linear FI stuff. Computes a transfer index.
del_fi = linear_fi_array - baseline_linear_fi_array
max_diff = linear_fi_array.max(axis=1, keepdims=True) - baseline_linear_fi_array
plt.figure(figsize=(4, 3))
plt.plot((del_fi / max_diff)[0])
# plt.plot((del_fi / max_diff)[10])
# plt.plot((del_fi / max_diff)[20])
plt.axhline(0)
plt.ylim(-0.2, 1)
# plt.savefig('figures/2l_specificity.svg')

#%% Compute overlap of the linear subnetwork
probe_args = copy.copy(args)

l1_overlap = np.zeros((len(sig_w_array), len(probe_angles)))
l2_overlap = np.zeros((len(sig_w_array), len(probe_angles)))

l1_width = np.zeros((len(sig_w_array), len(probe_angles)))
l2_width = np.zeros((len(sig_w_array), len(probe_angles)))

l1_bl_width = np.zeros_like(l1_width)
l2_bl_width = np.zeros_like(l1_width)
for sig_w_ind in trange(len(sig_w_array), position=0):

    probe_args.sig_w = sig_w_array[sig_w_ind]
    _net = model.Model(probe_args)

    t_utils.load_theory_changes(_net, [delw1_list[sig_w_ind], delw2_list[sig_w_ind]], active_inds_list[sig_w_ind])

    probe_args.theta = np.pi
    probe_stim = train_utils.GaborStimuli(probe_args, simple_mode=True, verbose=False)

    train_active_inds = r_utils.get_effective_weights(_net, probe_stim.x0)[1]

    l1_bl_width[sig_w_ind] = len(train_active_inds[1])
    l2_bl_width[sig_w_ind] = len(train_active_inds[2])
    for j in range(len(probe_angles)):
        probe_args.theta = probe_angles[j]

        probe_stim = train_utils.GaborStimuli(probe_args, simple_mode=True, verbose=False)
        test_active_inds = r_utils.get_effective_weights(_net, probe_stim.x0)[1]

        l1_overlap[sig_w_ind, j] = len(np.intersect1d(test_active_inds[1], train_active_inds[1])) / np.sqrt(len(test_active_inds[1]) * len(train_active_inds[1]))
        l1_width[sig_w_ind, j] = len(test_active_inds[1])
        l2_overlap[sig_w_ind, j] = len(np.intersect1d(test_active_inds[2], train_active_inds[2])) / np.sqrt(len(test_active_inds[1]) * len(train_active_inds[1]))
        l2_width[sig_w_ind, j] = len(test_active_inds[2])
#%%
plt.figure()
plt.imshow(l1_overlap); plt.colorbar()
plt.figure()
plt.imshow(l2_overlap); plt.colorbar()

delfi = linear_fi_array / baseline_linear_fi_array
delfi -= delfi.mean(axis=1, keepdims=True)
plt.figure()
plt.scatter(l1_overlap.flatten(), np.abs(delfi).flatten())

#%%
plt.figure()
plt.scatter((l1_width / l1_bl_width).flatten(), (linear_fi_array / baseline_linear_fi_array).flatten()); plt.colorbar() 

plt.figure()
plt.imshow(l2_width / l2_bl_width); plt.colorbar() 
#%% Plot transfer results

r_utils.fi_transfer_fig(linear_fi_array, baseline_linear_fi_array)
r_utils.error_transfer_fig(linear_fi_array, baseline_linear_fi_array)

#%%
plt.figure()

for i in [0, 10, 20]:
    _u, _s, _v = np.linalg.svd(delw1_list[i])
    plt.plot(_u[:, 0], color=str(i / 25), label=f"$\sigma_w=${sig_w_array[i]:.3f}")
plt.legend(); plt.grid(); plt.title('$\Delta W_1$ left vector, 2 Layers')

plt.figure()

for i in [0, 10, 20]:
    _u, _s, _v = np.linalg.svd(delw1_list[i])
    plt.plot(_v[0, :], color=str(i / 25), label=f"$\sigma_w=${sig_w_array[i]:.3f}")
plt.legend(); plt.grid(); plt.title('$\Delta W_1$ right vector, 2 Layers')

#%%
plt.figure()

for i in [0, 10, 20]:
    _u, _s, _v = np.linalg.svd(delw2_list[i])
    plt.plot(_u[:, 0], color=str(i / 25), label=f"$\sigma_w=${sig_w_array[i]:.3f}")
plt.legend(); plt.grid(); plt.title('$\Delta W_2$ left vector, 2 Layers')

plt.figure()

for i in [0, 10, 20]:
    _u, _s, _v = np.linalg.svd(delw2_list[i])
    plt.plot(_v[0, :], color=str(i / 25), label=f"$\sigma_w=${sig_w_array[i]:.3f}")
plt.legend(); plt.grid(); plt.title('$\Delta W_2$ right vector, 2 Layers')

#%%
ind = 10
layer_ind = 0
plt.figure()
cov = response_properties[layer_ind]['covar'][ind]
corr = cov / np.sqrt(np.diag(cov).reshape(-1, 1) @ np.diag(cov).reshape(1, -1))
plt.hist(corr.flatten(), bins=20, alpha=0.5)
plt.axvline(corr[np.isnan(corr)==False].mean())

cov = response_properties[layer_ind]['covar0'][ind]
corr = cov / np.sqrt(np.diag(cov).reshape(-1, 1) @ np.diag(cov).reshape(1, -1))
plt.hist(corr.flatten(), bins=20, alpha=0.5)
plt.axvline(corr[np.isnan(corr)==False].mean(), color='C1', ls='--')

#%% Get information-limiting correlation results
theory_ind = 0
layer_ind = 1
args2 = copy.copy(args)
args2.sig_w = sig_w_array[theory_ind]
net2 = model.Model(args2)

stim = train_utils.GaborStimuli(args2, simple_mode=True, verbose=False)

W_effs_bl = r_utils.get_effective_weights(net2, stim.x0)[0]

t_utils.load_theory_changes(net2,
[delw1_list[theory_ind], delw2_list[theory_ind]], active_inds_list[theory_ind])

W_effs = r_utils.get_effective_weights(net2, stim.x0)[0]

if layer_ind == 1:
    forward_mat = W_effs[0]
    forward_mat_bl = W_effs_bl[0]
elif layer_ind == 2:
    forward_mat = W_effs[1] @ W_effs[0]
    forward_mat_bl = W_effs_bl[1] @ W_effs_bl[0]

sampling_ratios = np.linspace(0, 1, 50)
n_samples_per_value = 10
downsample_info = np.zeros((len(sampling_ratios), n_samples_per_value))
downsample_info_bl = np.zeros((len(sampling_ratios), n_samples_per_value))

for i in trange(len(sampling_ratios)):
    n_to_sample = int(forward_mat.shape[0] * sampling_ratios[i])
    for j in range(n_samples_per_value):
        random_inds = np.random.permutation(np.arange(forward_mat.shape[0]))
        sampled_forward_mat = forward_mat[random_inds[:n_to_sample]]
        sampled_forward_mat_bl = forward_mat_bl[random_inds[:n_to_sample]]
        
        signal = sampled_forward_mat @ standard_stim.x1.t().numpy()
        downsample_info[i, j] = signal.T @ np.linalg.inv(sampled_forward_mat @ sampled_forward_mat.T + 1e-7 * np.eye(n_to_sample)) @ signal

        signal_bl = sampled_forward_mat_bl @ standard_stim.x1.t().numpy()
        downsample_info_bl[i, j] = signal_bl.T @ np.linalg.inv(sampled_forward_mat_bl @ sampled_forward_mat_bl.T + 1e-7 * np.eye(n_to_sample)) @ signal_bl

# Plot information-limiting correlation results
plt.figure()
plt.errorbar(sampling_ratios, downsample_info.mean(1), downsample_info.std(1))
plt.axhline(downsample_info.mean(1)[-1])
plt.errorbar(sampling_ratios, downsample_info_bl.mean(1), downsample_info_bl.std(1))
plt.axhline(downsample_info_bl.mean(1)[-1])
#%% Plot tuning curves
theory_ind = -1
args2 = copy.copy(args)
args2.sig_w = sig_w_array[theory_ind]
net2 = model.Model(args2)
response_mat0 = r_utils.get_response_mats(net2, 2, 0.3, args2.N)

t_utils.load_theory_changes(net2, [delw1_list[theory_ind], delw2_list[theory_ind]], active_inds_list[theory_ind])

response_mat = r_utils.get_response_mats(net2, 2, 0.3, args2.N)

neuron_ind = 300
plt.figure()
plt.plot(response_mat0[:, neuron_ind])
plt.plot(response_mat[:, neuron_ind])
