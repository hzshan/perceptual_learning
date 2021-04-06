#%%
import matplotlib.pyplot as plt
import numpy as np
import train_utils, glob, torch, pickle, utils, model, copy, warnings
from tqdm import trange
import response_utils as r_utils
import theory_utils as t_utils

'''
Updated 8/19/2020. This assumes both a and delW to change.

For a comparison with simulations results, make sure the task parameters match. 
'''

%load_ext autoreload
%autoreload 2

parser = train_utils.Args('1D Gabor')
parser.add('nonlinearity', 'relu')
parser.add('loss', 'MSE') # MSE or BCE

# model parameters
parser.add('N', 1000); parser.add('Nhid', 1000); parser.add('n_layers', 1)

# task parameters
parser.add('sig_w', 0.2); parser.add('sig_s', 0.6); parser.add('theta', np.pi)
parser.add('noise_var', 0.01)

# training parameter
parser.add('eta', 1e-4)
parser.add('n_learn', 2000000)
parser.add('n_train_trials', 500)
parser.add('n_test_trials', 10000) 
parser.add('test_interval', 500)

args = parser.parse_args()
standard_stim = train_utils.GaborStimuli(args, simple_mode=False)
standard_net = model.Model(args)


def mse_optimal_a_wrong(forward_mat, stimuli, sing_val_truncation=1):
    """
    Wrong formula. For debugging only.
    """
    warnings.warn('Using a wrong formula for initial readout.')
    x1 = stimuli.x1_normed.t().cpu().numpy()
    F_u, F_s, F_v = np.linalg.svd(forward_mat)

    '''Calculate a coefficient'''
    xvsq = np.linalg.norm(F_v[:sing_val_truncation] @ x1)**2
    delta = np.linalg.norm(stimuli.x1.cpu().numpy()) # this delta is not delta theta
    coef = (1 + delta**2 / stimuli.noise_var) / (1 + delta**2 / stimuli.noise_var * xvsq)

    '''Compute the direction'''
    inv_noise = np.linalg.pinv(forward_mat @ forward_mat.T, rcond=(F_s**2)[sing_val_truncation] / (F_s**2)[0] - 0.001)
    a = inv_noise @ forward_mat @ x1
    return a * coef



#%% Solve self-consistent equations to get dela, delW
k = 1  # controls relative strength of L2 constraint on W and a. k=1 means equal strength.
def self_consistent(u, v, W1, a, k, x1):
    Lambda = np.linalg.inv(u*np.eye(W1.shape[1]) + (k - v)**-1 * W1.T @ W1) @ (x1 - (1-v/k)**-1 * W1.T @ a)
    Dela = (k - v)**-1 * (W1 @ Lambda + v * a)
    return np.linalg.norm(a+Dela)**2, np.linalg.norm(Lambda)**2, Lambda, Dela


update_coef = 0.9  # How quickly to update order parameters in the solver. 
#Should be between \geq 0 and < 1. Larger value means slower updates. 

sig_w_array = np.linspace(0.1, 1.0, 30) # range of sig_w to solve for

dela_over_a = np.zeros(len(sig_w_array))
delw_norm = np.zeros(len(sig_w_array))

u_arr = np.zeros(len(sig_w_array))
v_arr = np.zeros(len(sig_w_array)) 

a_norms = np.zeros(len(sig_w_array))

probe_args = copy.copy(args)

Dela_list = []
a_list = []
delw1_list = []
active_inds_list = []
OPs = []

for i in trange(len(sig_w_array)):
    probe_args.sig_w = sig_w_array[i]; probe_net = model.Model(probe_args)
    
    W_effs, active_inds = r_utils.get_effective_weights(probe_net, standard_stim.x0)
    W1 = W_effs[0]

    a = train_utils.mse_optimal_a(W1, standard_stim, sing_val_truncation=3)

    x1 = standard_stim.x1_normed.t().numpy()

    # initial guesses of order parameters
    u_traj = [10]
    v_traj = [0.0002]

    # iterate the self-consistent equations to solve them
    for j in range(300):
        u_new, v_new, Lambda, Dela = self_consistent(u_traj[-1], v_traj[-1], W1, a, k, x1)
        u_traj.append(u_new * (1-update_coef) + u_traj[-1] * update_coef)
        v_traj.append(v_new * (1-update_coef) + v_traj[-1] * update_coef)
    
    dela_over_a[i] = np.linalg.norm(Dela) / np.linalg.norm(a)

    delw_norm[i] = np.linalg.norm(a @ Lambda.T)
    u_arr[i] = u_new
    v_arr[i] = v_new
    a_norms[i] = np.linalg.norm(a)
    OPs.append([u_new, v_new])

    a_list.append(a)
    Dela_list.append(Dela)
    delw1_list.append(a @ Lambda.T)
    active_inds_list.append(active_inds)

#%% Save theory results

file_name = 'Saved Results/theory_1L_sigs0P6'

pickle.dump({'delw1':delw1_list, 'OP':OPs, 'a':a_list, 'sig_w':sig_w_array, 'active_inds':active_inds_list, 'dela_list':Dela_list}, open(file_name, 'wb'))

#%%
plt.figure()
u, s, v = np.linalg.svd(W1 @ W1.T)
plt.plot(np.log10(s), marker='o')
#%% Check theory results (for convergence etc.)
plt.figure()
plt.plot(v_traj)

plt.figure()
plt.plot(delw_norm)

plt.figure()
plt.plot(dela_over_a)

plt.figure()
plt.plot(a_norms)

#%%
plt.plot()

#%% Load results from simulations
name = '1L_MSE_GD_sigw_sigs0p2_etaP00001_'

utils.process_raw_data(name)  # This will overwrite the data file. Make sure this line is usually commented out.

dataset = utils.find_and_load_data(name, 'sig_w')

dela_over_a_data = []
delw_data = []
sig_w_data = []
sing_val_ratio = []

for data in dataset:
    sig_w_data.append(data['args'].sig_w)
    dela_over_a_data.append(np.linalg.norm(data['a'][-1] - data['a'][0]) / np.linalg.norm(data['a'][0]))
    delw_data.append(np.linalg.norm(utils.recover_matrix(data['delW'][0][-1])))
    sing_val_ratio.append(data['delW'][0][-1]['s'][1] / data['delW'][0][-1]['s'][0])

plt.figure()
plt.scatter(sig_w_data, delw_data)
#%% Compare theory and simulation

plt.figure()
plt.scatter(np.sort(sig_w_data), np.array(dela_over_a_data)[np.argsort(sig_w_data)], label='Simulation', color='C1')

plt.plot(sig_w_array, dela_over_a, label='Theory')
plt.xlabel('$\sigma_w$')
plt.title('$|\Delta a|/|a_0|$')
plt.savefig('figures/raw/dela_1l.svg')
# plt.legend()


plt.figure()
plt.plot(sig_w_array, delw_norm, label='Theory')
plt.scatter(np.sort(sig_w_data), np.array(delw_data)[np.argsort(sig_w_data)], label='Simulation', color='C1')
# plt.scatter(np.sort(sig_w_data), np.array(dela_over_a_data)[np.argsort(sig_w_data)], label='Simulation', color='C1')

plt.xlabel('$\sigma_w$')
plt.title('$|\Delta W|/|W_0|$')
plt.savefig('figures/raw/delW_1l.svg')
# plt.legend()

#%% svg, second singular value over the first one
plt.figure()
plt.scatter(np.sort(sig_w_data), np.array(sing_val_ratio)[np.argsort(sig_w_data)][:12])

#%% Redistributing changes between a and W
# k_array = np.array([1e-3])
k_array = np.array([1e-4, 1e-3, 1e-2, 1e-1, 1, 10, 100, 1000])


dela_over_a = np.zeros(len(k_array))
delw = np.zeros(len(k_array))

u_arr = np.zeros(len(k_array))
v_arr = np.zeros(len(k_array)) 

a_norms = np.zeros(len(k_array))

probe_args = copy.copy(args)

errors_list = []

for i in trange(len(k_array), position=0):
    probe_args.sig_w = 0.1
    probe_net = model.Model(probe_args)
    W1= r_utils.get_effective_weights(probe_net, standard_stim.x0)[0][0]

    a = train_utils.mse_optimal_a(W1, standard_stim)

    x1 = standard_stim.x1_normed.t().numpy()

    u_traj = [25]
    v_traj = [0.0005]

    for j in range(100):
        u_new, v_new, Lambda, Dela = self_consistent(u_traj[-1], v_traj[-1], W1, a, k_array[i], x1)
        u_traj.append(u_new)
        v_traj.append(v_new)
    
    dela_over_a[i] = np.linalg.norm(Dela) / np.linalg.norm(a)

    delw[i] = np.linalg.norm((a) @ Lambda.T)
    u_arr[i] = u_new
    v_arr[i] = v_new
    a_norms[i] = np.linalg.norm(a)

    transfer_net = model.Model(args)
    angles, errors0 = train_utils.test_at_angles_meanfield(transfer_net, args, npoints=32)

    active_inds = r_utils.get_active_inds(transfer_net, standard_stim.x0)
    full_delw = np.zeros_like(transfer_net.Ws[0])
    full_delw[active_inds[1]] = (a+Dela) @ Lambda.T
    transfer_net.RO.weight.data = torch.zeros_like(transfer_net.RO.weight)
    transfer_net.RO.weight.data.numpy()[0, active_inds[1]] = (a + Dela).flatten()
    transfer_net.Ws[0] += full_delw

    angles, errors = train_utils.test_at_angles_meanfield(transfer_net, args, npoints=32)
    errors_list.append(errors)

#%%
plt.figure()
plt.plot(np.log10(k_array), delw, marker='o')
plt.xlabel('log10(k)')
plt.title('$\\frac{|\Delta W|}{|W|}$')
plt.figure()
plt.plot(np.log10(k_array), np.log10(dela_over_a), marker='o')
plt.xlabel('log10(k)')
plt.title('$\log_{10} \\frac{|\Delta a|}{|a|}$')
#%%
v_eff = (W1.T + Lambda @ (a+Dela).T) @ (a+Dela)
plt.figure()
plt.plot(a)
plt.plot(Dela)

plt.figure()
plt.imshow((a+Dela) @ Lambda.T)

plt.figure()
plt.plot(v_eff)
plt.plot(standard_stim.x1_normed.t())

#%% Plot order parameters

plt.figure()
plt.plot(u_arr)

plt.figure()
plt.plot(v_arr)

#%%
plt.figure()

for i, err in enumerate(errors_list):
    if i < 1:
        continue
    plt.plot(angles, err, label=f'log10k={np.log10(k_array[i])}')
plt.axhline(errors0.mean())
plt.xlabel('Stimulus')
plt.ylabel('Error')
plt.legend(loc=5)

#%%

data_stim = train_utils.GaborStimuli(data_args, simple_mode=True)
length = len(dataset[1]['theta1'])
data = dataset[1]
data_args = data['args']
plt.figure()
plt.scatter(range(len(dataset[1]['theta1'])), dataset[1]['theta1'])
plt.scatter(range(len(dataset[1]['theta1'])), dataset[1]['theta2'])

theory_theta1 = np.zeros(len(dataset[1]['theta1']))
theory_theta2 = np.zeros_like(theory_theta1)
theory_theta1[0] = dataset[1]['theta1'][0]
theory_theta2[0] = dataset[1]['theta2'][0]

def theta1_func(theta10, time):
    return 1 - (1 - theta10) * np.exp(-time * (np.linalg.norm(data['a'][0])**2 * (1 + data_args.noise_var)) * 2 * np.linalg.norm(data_stim.x1))

def theta2_func(theta20, time):
    return theta20 * np.exp(-time * np.linalg.norm(data['a'][0])**2 * data_args.noise_var * 2)

time_axis = np.linspace(0, length, 100)

plt.plot(time_axis, theta1_func(data['theta1'][0], time_axis * data_args.test_interval * data_args.eta))
plt.plot(time_axis, theta2_func(data['theta2'][0], time_axis * data_args.test_interval * data_args.eta))
plt.savefig('figures/order_params.svg')

# %% Slope

sim_ind = 23
raw_slope = np.abs(probe_net.Ws[0] @ standard_stim.x1_normed.t().numpy())
mask = np.zeros_like(raw_slope)
mask[active_inds_list[sim_ind][1]] = 1
slope = raw_slope * mask

raw_slope_naive = np.abs(probe_net.init_weights[0].numpy() @ standard_stim.x1_normed.t().numpy())
mask = np.zeros_like(raw_slope)
mask[active_inds_list[sim_ind][1]] = 1
slope = raw_slope * mask

plt.figure()
plt.plot(slope)