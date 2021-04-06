"""
Temporary file for checking simulations
"""
#%%
import matplotlib.pyplot as plt
import numpy as np
import train_utils, torch, pickle, utils, model, copy, warnings
from tqdm import trange, tqdm
import response_utils as r_utils
import theory_utils as t_utils

"""Keywords for quick access with Search
DELCOM: compare delW norm between theory and simulation
PLTDELW: imshow() delW matrices
RESPROP: get response properties
"""

theory_dict = pickle.load(open('Saved Results/theory_3L_sigs0P2', 'rb'))
delw1_list = theory_dict['delw1']; delw2_list = theory_dict['delw2']
delw3_list = theory_dict['delw3']

dw1_norm = [np.linalg.norm(_m) for _m in delw1_list]
dw2_norm = [np.linalg.norm(_m) for _m in delw2_list]
dw3_norm = [np.linalg.norm(_m) for _m in delw3_list]
sig_w_array = np.linspace(0.1, 1, 30)
#%% load a simulation
filename = '3L_MSE_GD_sigw_sigs0p2_lambdaP015_iter60M_'
utils.process_raw_data(filename)
dataset = utils.find_and_load_data(filename, attribute='sig_w')

print('Nominal eta', dataset[0]['args'].eta * np.linalg.norm(dataset[0]['a'][-1]))
def get_delw_norms_from_dataset(dataset):
    data_sigw = []
    data_delw1_norm = []
    data_delw2_norm = []
    data_delw3_norm = []
    for _data in dataset:
        data_sigw.append(_data['args'].sig_w)
        _net = model.Model(_data['args'])
        data_delw1_norm.append(np.linalg.norm(utils.recover_matrix(_data['delW'][0][-1])))
        data_delw2_norm.append(np.linalg.norm(utils.recover_matrix(_data['delW'][1][-1])))
        data_delw3_norm.append(np.linalg.norm(utils.recover_matrix(_data['delW'][2][-1])))
    return np.sort(data_sigw), \
    np.array(data_delw1_norm)[np.argsort(data_sigw)], \
    np.array(data_delw2_norm)[np.argsort(data_sigw)], \
    np.array(data_delw3_norm)[np.argsort(data_sigw)]

data_sigw_l2, data_delw1_l2, data_delw2_l2, data_delw3_l2 = get_delw_norms_from_dataset(dataset)

plt.figure()
plt.plot(sig_w_array, dw1_norm); plt.plot(sig_w_array, dw2_norm); plt.plot(sig_w_array, dw3_norm)
plt.scatter(data_sigw_l2, data_delw1_l2)
plt.scatter(data_sigw_l2, data_delw2_l2)
plt.scatter(data_sigw_l2, data_delw3_l2)

#%% DELTRAJ Do some validation analysis 


plt.figure()
sim_ind = 2
delw3_traj = [np.linalg.norm(d['s']) for d in dataset[sim_ind]['delW'][2]]

print(dataset[sim_ind]['args']); print('nominal eta:', np.linalg.norm(dataset[sim_ind]['a'][-1]) * dataset[sim_ind]['args'].eta)
plt.plot(delw3_traj); plt.axhline(delw3_traj[-1]); plt.axhline(dw3_norm[-1], color='r', label='theory')
plt.title(dataset[sim_ind]['args'].sig_w)
plt.xlabel(f'*{dataset[sim_ind]["args"].test_interval}')

plt.figure()
plt.imshow(utils.recover_matrix(dataset[sim_ind]['delW'][-1][-1]))

# Compare x0
plt.figure()
_net = model.Model(dataset[sim_ind]['args'])
_stim = train_utils.GaborStimuli(dataset[sim_ind]['args'], simple_mode=True, verbose=False)
plt.plot(_net.get_obs(_stim.x0).t().data)
utils.load_GD_changes(_net, dataset[sim_ind], time_index=-150)
plt.plot(_net.get_obs(_stim.x0).t().data)
plt.title('Compare x0')

plt.figure()
plt.plot(np.log10(dataset[sim_ind]['loss']))

#%% temp viewer

temp = pickle.load(open(os.getcwd() + '/Raw_results/3L_MSE_GD_sigw_sigs0p2_lambdaP015_/1.results', 'rb'))
temp['delW'][-1]['layer_0']


#%% PLOT ACROSS LAMBDA
filename = '3L_MSE_GD_lambda_sigs0p2_sigw0p8_'
utils.process_raw_data(filename)
dataset = utils.find_and_load_data(filename, attribute='lambda2')

data_lambda = []
loss = []
for _data in dataset:
    data_lambda.append(_data['args'].lambda2)
    loss.append(np.log(_data['loss'][-1]))

plt.figure()
plt.scatter(data_lambda, loss)

