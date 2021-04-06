#%%
import matplotlib.pyplot as plt
import numpy as np
import train_utils, torch, pickle, utils, model, copy, warnings, time, os
import response_utils as r_utils
import theory_utils as t_utils
"""
Search solutions with fsolve, using results from the iterative solver as initial conditions.
Created 8/14/2020.
"""

def scatter_simulation_results(file_name):
    dataset = utils.find_and_load_data(file_name, attribute='sig_w')

    data_sigw = []; data_delw1_norm = []; data_delw2_norm = []; data_delw3_norm = []
    for _data in dataset:
        data_sigw.append(_data['args'].sig_w)
        data_delw1_norm.append(np.linalg.norm(utils.recover_matrix(_data['delW'][0][-1])))
        data_delw2_norm.append(np.linalg.norm(utils.recover_matrix(_data['delW'][1][-1])))
        data_delw3_norm.append(np.linalg.norm(utils.recover_matrix(_data['delW'][2][-1])))

    plt.scatter(np.sort(data_sigw), np.array(data_delw1_norm)[np.argsort(data_sigw)], marker='o')
    plt.scatter(np.sort(data_sigw), np.array(data_delw2_norm)[np.argsort(data_sigw)], marker='o')
    plt.scatter(np.sort(data_sigw), np.array(data_delw3_norm)[np.argsort(data_sigw)], marker='o')

sig_w_array = np.linspace(0.1, 1, 30)

# Load results from the iterative solver. Parameters: update coef=0.99
raw_result_dir = os.getcwd() + '/theory_raw_results/'
a_list = []; OPs_trajs = []; final_OPs = np.zeros((30, 4))


for i in range(30):
    _data = pickle.load(open(raw_result_dir + str(i+1), 'rb'))
    a_list.append(_data['a'])
    OPs_trajs.append(np.array(_data['OP_traj']))
    final_OPs[i] = _data['OP_traj'][-1]

#%% Load a simulation
utils.process_raw_data('3Layers_SquaredError_GradDescent_sigw_sigs0p2_lambda2_p1_')
dataset = utils.find_and_load_data('3Layers_SquaredError_MeanField_sigw_sigs0p2_lambda2_p5_', attribute='sig_w')

#%% Loss in the simulations



plt.figure()
data_loss = []
for _data in dataset:
    data_loss.append(_data['loss'][-1])

plt.figure()
plt.scatter(data_sigw, np.log10(data_loss))

#%% dela in the simulations

plt.figure()
dela_norms = []
for _data in dataset:
    dela_norms.append(np.linalg.norm(_data['a'][-1] - _data['a'][0]))

plt.scatter(data_sigw, dela_norms)

#%% Plot loss over time for each trial
dataset = utils.find_and_load_data('3Layers_SquaredError_MeanField_sigw_sigs0p2_lambda2_2_', attribute='sig_w')

plt.figure()
plt.plot(np.log10(dataset[7]['loss']), label='0.1')
plt.plot(np.log10(dataset[2]['loss']), label='0.6')
plt.plot(np.log10(dataset[5]['loss']), label='1.0')
plt.ylabel('log10 loss')
plt.legend()
#%% Refine solutions with fsolve. Also check whether found solutions solve the equations.
from scipy.optimize import fsolve
from tqdm import trange

fsolve_OPs = []
fsolve_delw1 = []
fsolve_delw2 = []
fsolve_delw3 = []
active_inds_list = []
a_list = []

OPs_errors = []
fsolve_errors = []
a_norms = []

delw1_list = []; delw2_list = []; delw3_list = []

for theory_ind in trange(len(sig_w_array)):

    args.sig_w = sig_w_array[theory_ind]
    _net = model.Model(args)

    # first use full Ws to get the same a as in the simulations
    W_effs, active_inds = r_utils.get_effective_weights(_net, stim.x0, full_mat=True)
    W1, W2, W3 = W_effs; active_inds_list.append(active_inds)
    
    a = train_utils.mse_optimal_a(W3 @ W2 @ W1, stim)[active_inds[-1]].reshape(-1, 1)
    a_list.append(a)
    a_norm_sq = np.linalg.norm(a)**2
    a_norms.append(np.linalg.norm(a))

    W1, W2, W3 = r_utils.get_effective_weights(_net, stim.x0)[0]

    def norm_sq(x):
        return np.linalg.norm(x)**2


    def self_consistent_eqs(p):
        u1, u2, v1, v2 = p
        one_minus_v1_u2 = 1 - v1 * u2
        one_minus_a_norm_sq_v2 = 1 - a_norm_sq * v2

        Q1 = u1 * np.eye(W1.shape[1]) + one_minus_v1_u2**-1 * u2 * W1.T @ W1
        Q2 = one_minus_a_norm_sq_v2*np.eye(W2.shape[0])-a_norm_sq*one_minus_v1_u2**-1*v1*W2 @ W2.T

        invQ2 = np.linalg.inv(Q2)

        inv_mat = Q1 + a_norm_sq*one_minus_v1_u2**-2 * W1.T @ W2.T @ invQ2 @ W2 @ W1
        _u, _s, _v = np.linalg.svd(inv_mat)
        # Lambda = np.linalg.inv(Q1 + a_norm_sq*one_minus_v1_u2**-2 * W1.T @ W2.T @ invQ2 @ W2 @ W1) @ (x1 - one_minus_v1_u2**-1 * W1.T @ W2.T @ invQ2 @ W3.T @ a)

        Lambda = np.linalg.inv(inv_mat) @ (x1 - one_minus_v1_u2**-1 * W1.T @ W2.T @ invQ2 @ W3.T @ a)

        U2 = invQ2 @ (W3.T @ a + a_norm_sq * one_minus_v1_u2**-1 * W2 @ W1 @ Lambda)
        
        u1_out = norm_sq(one_minus_v1_u2**-1 * (W2.T @ U2 + u2 * W1 @ Lambda))
        u2_out = norm_sq(U2)
        v1_out = norm_sq(Lambda)
        v2_out = norm_sq(one_minus_v1_u2**-1 * (W1 @ Lambda + v1 * W2.T @ U2))
        return (u1_out - u1) / u1_out, (u2_out - u2) / u2_out, (v1_out - v1) / v1_out, (v2_out - v2) / v2_out

    fsolve_OP = fsolve(self_consistent_eqs, final_OPs[theory_ind], factor=0.5, maxfev=100000)

    fsolve_OPs.append(fsolve_OP)

    # compute delW norms with fsolve parameters
    delW1, delW2, delW3 = t_utils.get_del_Ws_threelayer(*fsolve_OP, W1, W2, W3, a, a_norm_sq, x1)
    fsolve_delw1.append(np.linalg.norm(delW1))
    fsolve_delw2.append(np.linalg.norm(delW2))
    fsolve_delw3.append(np.linalg.norm(delW3))
    delw1_list.append(delW1)
    delw2_list.append(delW2)
    delw3_list.append(delW3)

    fsolve_errors.append(self_consistent_eqs(fsolve_OP))
    OPs_errors.append(self_consistent_eqs(final_OPs[theory_ind]))

fsolve_OPs = np.array(fsolve_OPs)

# save it 
result_dict = {'OP':fsolve_OPs, 'delw1':delw1_list, 'delw2':delw2_list, 'delw3':delw3_list, 'active_inds':active_inds_list, 'a':a_list}

#%%
pickle.dump(result_dict, open('Saved Results/THEORY_RESULTS_3_layers_sigs0p2', 'wb'))
#%% ###### Compare theory results with simulation
plt.figure()
plt.plot(sig_w_array, np.array(fsolve_delw1))
plt.plot(sig_w_array, np.array(fsolve_delw2))
plt.plot(sig_w_array, np.array(fsolve_delw3))

name = '3Layers_SquaredError_MeanField_sigw_sigs0p2_lambda2_p25_'

scatter_simulation_results(name)

#%% Plot v_eff difference from simulations

dataset = utils.find_and_load_data('3Layers_SquaredError_MeanField_sigw_sigs0p2_lambda2_p04_small_', attribute='sig_w')

data_err = []; data_veff_cos = []
for _data in dataset:
    data_veff_cos.append(utils.cos(_data['v_eff'][-1], x1))

plt.figure()
plt.scatter(data_sigw, data_veff_cos)
#%%
dataset = utils.find_and_load_data('3Layers_SquaredError_MeanField_sigw_sigs0p2_lambda2_p5_', attribute='sig_w')
def estimate_OP_from_data(data):
    args = data['args']
    net = model.Model(args)
    stim = train_utils.GaborStimuli(args, simple_mode=True, verbose=False)
    W_effs, active_inds = r_utils.get_effective_weights(net, stim.x0)
    W1, W2, W3 = W_effs
    # print(W1.shape)
    delw1 = utils.recover_matrix(data['delW'][0][-1])[active_inds[1], :]
    delw2 = utils.recover_matrix(data['delW'][1][-1])[active_inds[2], :][:, active_inds[1]]
    delw3 = utils.recover_matrix(data['delW'][2][-1])[active_inds[3], :][:, active_inds[2]]
    a = data['a'][-1][active_inds[3]].reshape(-1, 1)
    a_norm_sq = np.linalg.norm(a)**2
    u1 = np.linalg.norm((delw2 + W2).T @ (delw3 + W3).T @ a)**2
    u2 = np.linalg.norm((delw3 + W3).T @ a)**2
    v1 = np.linalg.norm(delw1)**2 / u1
    v2 = np.linalg.norm(delw2)**2 / u2
    print(u1, u2, v1, v2)

    one_minus_v1_u2 = 1 - v1 * u2
    one_minus_a_norm_sq_v2 = 1 - a_norm_sq * v2

    Q1 = u1 * np.eye(W1.shape[1]) + one_minus_v1_u2**-1 * u2 * W1.T @ W1
    Q2 = one_minus_a_norm_sq_v2*np.eye(W2.shape[0])-a_norm_sq*one_minus_v1_u2**-1*v1*W2 @ W2.T

    invQ2 = np.linalg.inv(Q2)

    inv_mat = Q1 + a_norm_sq*one_minus_v1_u2**-2 * W1.T @ W2.T @ invQ2 @ W2 @ W1

    Lambda = np.linalg.inv(inv_mat) @ (x1 - one_minus_v1_u2**-1 * W1.T @ W2.T @ invQ2 @ W3.T @ a)

    U2 = invQ2 @ (W3.T @ a + a_norm_sq * one_minus_v1_u2**-1 * W2 @ W1 @ Lambda)
    
    u1_out = norm_sq(one_minus_v1_u2**-1 * (W2.T @ U2 + u2 * W1 @ Lambda))
    u2_out = norm_sq(U2)
    v1_out = norm_sq(Lambda)
    v2_out = norm_sq(one_minus_v1_u2**-1 * (W1 @ Lambda + v1 * W2.T @ U2))
    return (u1_out - u1) / u1_out, (u2_out - u2) / u2_out, (v1_out - v1) / v1_out, (v2_out - v2) / v2_out

estimate_OP_from_data(dataset[0])

#%%
args2 = copy.copy(args)
args2.N = 2000
stim2 = train_utils.GaborStimuli(args2, simple_mode=True)
v_eff_cos = [utils.cos(vec, stim2.x1.numpy()) for vec in dataset[0]['v_eff']]
plt.figure()
plt.plot(v_eff_cos)

#%%
plt.figure()
delw1_norms = [np.linalg.norm(d['s']) for d in dataset[0]['delW'][0]]
plt.plot(delw1_norms)