#%%
import numpy as np
import train_utils, torch, pickle, utils, model, copy, warnings, time, os
import response_utils as r_utils
import theory_utils as t_utils

"""
Created Aug 14, 2020
Solves the three layer theory in the narrow stimulus regime for one parameter. Used for parallel computing.
Dependency: PL/raw_op_file.pkl, which should be a list of final parameters from local simulations.
"""

torch.set_grad_enabled(False)

parser = train_utils.Args('1D Gabor')
parser.add('nonlinearity', 'relu')
parser.add('loss', 'MSE') # MSE or BCE

# model parameters
parser.add('N', 300); parser.add('Nhid', 300); parser.add('n_layers', 3)

# task parameters
parser.add('sig_w', 1.2); parser.add('sig_s', 0.2); parser.add('theta', np.pi)

# training parameter
parser.add('eta', 1e-4)
parser.add('n_learn', 2000000)
parser.add('n_train_trials', 500)
parser.add('n_test_trials', 10000) 
parser.add('test_interval', 500)
parser.add('noise_var', 1)

args = parser.parse_args()

result_dir = os.getcwd() + f'/Raw_results/{args.BATCH_NAME}'

raw_OP_dir = os.getcwd() + f'/raw_OPs_file.pkl'
raw_OPs = pickle.load(open(raw_OP_dir, 'rb'))

raw_sig_w = np.linspace(0.1, 1.0, 30)  # parameters used in the local simulations
best_match_ind = np.argmin((raw_sig_w - args.sig_w)**2)

init_OPs = raw_OPs[best_match_ind]

inv_ind = 3  # controls num of sing values to keep when inverting a matrix in the self-consistent equations

#%% USING a in-script solver (defined below)
average_if_not_converged = True
def predict_delw_threelayer(network:model.Model, stimuli, init_guess, update_coef, max_iter):
    """
    Solve the order parameter self-consistent equations for three-layer networks.
    """
    def norm_sq(x):
        return np.linalg.norm(x)**2

    x1 = stimuli.x1_normed.numpy().reshape(-1, 1)

    # create the effective weights
    w_effs, active_inds = r_utils.get_effective_weights(network, stimuli.x0); W1, W2, W3 = w_effs
    forward_mat = W3 @ W2 @ W1

    # _u, _s, _v = np.linalg.svd(forward_mat @ forward_mat.T)
    # a = np.linalg.pinv(forward_mat @ forward_mat.T, _s[1] / _s[0] - 0.01) @ forward_mat @ x1
    a = train_utils.mse_optimal_a(forward_mat, stimuli)
    
    a_norm_sq = np.linalg.norm(a)**2

    # Solve self-consistent equations by iterati\on
    def iteration_eqs(p):
        u1, u2, v1, v2 = p
        one_minus_v1_u2 = 1 - v1 * u2

        one_minus_a_norm_sq_v2 = 1 - a_norm_sq * v2

        if u1 < 10:
            u1 = 10       
        # if u1 > 1e3:
        #     u1 = 10
        # if u2 > 1e4:
        #     u2 = 10
        # if v1 > 1e-2:
        #     v1 = 1e-4
        # if v2 > 1e-3:
        #     v2 = 1e-5

        # stability control
        # if one_minus_a_norm_sq_v2 < 0:
        #     one_minus_a_norm_sq_v2 = 0.1
        
        # if one_minus_v1_u2 < 0:
        #     one_minus_v1_u2 = 0.1
        

        Q1 = u1 * np.eye(W1.shape[1]) + one_minus_v1_u2**-1 * u2 * W1.T @ W1
        Q2 = one_minus_a_norm_sq_v2*np.eye(W2.shape[0])-a_norm_sq*one_minus_v1_u2**-1*v1*W2 @ W2.T

        invQ2 = np.linalg.inv(Q2)

        inv_mat = Q1 + a_norm_sq*one_minus_v1_u2**-2 * W1.T @ W2.T @ invQ2 @ W2 @ W1
        _u, _s, _v = np.linalg.svd(inv_mat)
        # Lambda = np.linalg.inv(Q1 + a_norm_sq*one_minus_v1_u2**-2 * W1.T @ W2.T @ invQ2 @ W2 @ W1) @ (x1 - one_minus_v1_u2**-1 * W1.T @ W2.T @ invQ2 @ W3.T @ a)

        Lambda = np.linalg.pinv(inv_mat, rcond=_s[inv_ind] / _s[0] - 0.01) @ (x1 - one_minus_v1_u2**-1 * W1.T @ W2.T @ invQ2 @ W3.T @ a)

        U2 = invQ2 @ (W3.T @ a + a_norm_sq * one_minus_v1_u2**-1 * W2 @ W1 @ Lambda)
        
        u1_out = norm_sq(one_minus_v1_u2**-1 * (W2.T @ U2 + u2 * W1 @ Lambda))
        u2_out = norm_sq(U2)
        v1_out = norm_sq(Lambda)
        v2_out = norm_sq(one_minus_v1_u2**-1 * (W1 @ Lambda + v1 * W2.T @ U2))
        return np.array([u1_out, u2_out, v1_out, v2_out])

    op_vals = [np.array(init_guess)]  # initial guesses for the order parameters

    for i in range(max_iter):
        new_vals = iteration_eqs(op_vals[-1])

        convergence_counter = 0
        for old_op, new_op in zip(op_vals[-1], new_vals):
            if np.abs(new_op - old_op) / np.abs(old_op) < 1e-3:
                convergence_counter += 1

        if i == 0:
            op_vals.append(update_coef * op_vals[-1] + (1-update_coef) * new_vals)
        else:
            op_vals.append(update_coef * op_vals[-1] + (1-update_coef) * new_vals)

        if convergence_counter == 4:
            print('Converged')
            break
        
        # Heuristic divergence fuse: if value exceeds some range, raise RuntimeError
        if op_vals[-1][0] > 100:
            raise RuntimeError('Divergence detected because u1>100.')
        if op_vals[-1][1] > 5000:
            raise RuntimeError('Divergence detected because u2>1000.')

        # dampen oscillations
        if (i + 1) % 1000 == 0:
            # op_vals.append(np.mean(np.array(op_vals)[-1000:], axis=0))
            print(np.mean(np.array(op_vals)[-10:], axis=0))
    
    # if all iterations have been used, return an average
    if average_if_not_converged:
        if i == max_iter - 1:
            op_vals.append(np.mean(np.array(op_vals)[-10000:], axis=0))

    delW1, delW2, delW3 = t_utils.get_del_Ws_threelayer(*op_vals[-1], W1, W2, W3, a, a_norm_sq, x1)


    return np.array(op_vals), delW1, delW2, delW3, active_inds, a

#%%
update_coef = args.eta
max_iter = args.n_train_trials

tic = time.time()

net = model.Model(args)
stimuli = train_utils.GaborStimuli(args, simple_mode=True, verbose=False)

W_effs = r_utils.get_effective_weights(net, stimuli.x0)[0]

_OPs, delw1, delw2, delw3, w_bounds, a = predict_delw_threelayer(net, stimuli, init_guess=init_OPs, update_coef=update_coef, max_iter=max_iter)

output_dict = {'a':a, 'OP_traj':_OPs}
pickle.dump(output_dict, open(result_dir + '/' + args.BATCH_NAME + str(args.TRIAL_IND), 'wb'))