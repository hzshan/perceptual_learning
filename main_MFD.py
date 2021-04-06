#%%
import numpy as np
import torch, pickle, time, os, utils, model, train_utils, warnings
import matplotlib.pyplot as plt

# %load_ext autoreload
# %autoreload 2

"""
Ported from main_MFD.py as Mar 23
Converted to run the on the GPU.

A general note:
Make sure everything saved to the trial data file has been transferred to the cpu (a good way is to make sure that they are all numpy arrays. Otherwise one cannot open them on a computer without a CUDA configuration.)

Updates
- May 08: Fixed an issue with the MSE gradients
- June 02: Fixed an issue where the MSE loss is computed incorrectly

"""
if torch.cuda.is_available():
    gpu = torch.device('cuda:0')
else:
    gpu = None
    warnings.warn('GPU not found. Running on CPU.')


parser = train_utils.Args('1D Gabor')
parser.add('nonlinearity', 'relu')
parser.add('loss', 'MSE') # MSE or BCE

# model parameters
parser.add('N', 100); parser.add('Nhid', 100); parser.add('n_layers', 1)

# task parameters
parser.add('sig_w', 1.0); parser.add('sig_s', .2); parser.add('theta', np.pi)

# training parameter
parser.add('eta', 1e-5)
parser.add('n_learn', 200000)
parser.add('n_train_trials', 2000)
parser.add('n_test_trials', 2500)  # using memory-saving averages (20 replicas)
parser.add('test_interval', 10000)
parser.add('noise_var', 0.1)
parser.add('lambda2', 0.001)

args = parser.parse_args()

TIC = time.time()  # Start time of the script

mgr = train_utils.Manager(args, device=gpu); trainer = train_utils.Trainer(args); stimuli = train_utils.GaborStimuli(args, device=gpu)


net = model.Model(args, dims=[args.N, *[args.Nhid for i in range(args.n_layers)]])
net = net.to(device=gpu)


#%%

# get activity from each layer for the trained stimulus
all_obs = [net.get_obs(stimuli.x0, to_layer=i+1) for i in range(net.n_layers)]

# compute phi_prime vectors for all layers. These are explicitly fixed throughout learning
all_phi_prime = [(obs > 0).float().view(-1, 1) for obs in all_obs]

train_utils.pretrain_readout(network=net, manager=mgr, stimuli=stimuli, loss_type=args.loss, epsilon=1e-5)

# make the repeated phi_primes here.
repeated_phi_prime = [_phi_prime.cpu().repeat(1, net.dims[i]).to(gpu) for i, _phi_prime in enumerate(all_phi_prime)]

#%%
"""
Note:
All the list of vectors are in the order of L1, L2, ....

The vectors are so defined:
1. a_{eff}^l: \nabla_{h^l} r. Dimension: N_l by 1
2. m^l: \nabla_{x} x^l. Dimension: N_l by N_input.

Dont mess with these formulae!!! Last checked June 02
"""


trainer.start_trial(net, stimuli, args, mgr)

init_weights = [w.to(gpu) for w in net.init_weights]

net.RESET_WEIGHTS()

a = net.RO.weight.detach().t()

# gradients for weight matrices
W_grads = [None for i in range(args.n_layers)]

for iteration in range(args.n_learn):

    list_of_weights = [L.weight.detach() for L in net.Ls]

    # compute the current v_effective
    W_tildes = []
    for _repeated_phi_prime, _W in zip(repeated_phi_prime, list_of_weights):
        W_tildes.append(_repeated_phi_prime * _W)

    v_eff = a.clone()
    for w_tilde in W_tildes[::-1]:
        v_eff = w_tilde.T @ v_eff

    all_a_eff = [a for i in range(net.n_layers)]  # these will be first computed in the reverse order
    for i in range(net.n_layers):
        for j in range(i):
            all_a_eff[i] = (W_tildes[::-1][j].t() @ all_a_eff[i])
        all_a_eff[i] = all_phi_prime[::-1][i] * all_a_eff[i]

    all_a_eff.reverse()  # reverse such that the first element is a_eff 1

    m_matrices = [torch.eye(args.N).to(gpu) for i in range(net.n_layers)]
    for i in range(net.n_layers):
        for j in range(i):
            m_matrices[i] = W_tildes[j] @ m_matrices[i]

    forward_mat = torch.eye(args.N).to(gpu)
    for mat in W_tildes:
        forward_mat = mat @ forward_mat
    
    delta = torch.norm(stimuli.x1)

    # compute f1 and f2. These are shared by gradients across layers. For MSE, f1 is basically <r_+ - \hat{r_+}>, f2 is 1.
    if args.loss == 'BCE':
        # compute global coefficients numerically
        rm_fields = -stimuli.x1 @ v_eff + torch.normal(mean=torch.zeros(10000), std=np.sqrt(stimuli.noise_var)).to(gpu) * torch.norm(v_eff)
        f1 = torch.mean(torch.sigmoid(rm_fields))
        f2 = torch.mean(torch.sigmoid(rm_fields) - torch.sigmoid(rm_fields)**2)
        for i in range(args.n_layers):
            W_grads[i] = all_a_eff[i] @ (-f1 * m_matrices[i] @ stimuli.x1.t()).t() + (torch.ones(1) * stimuli.noise_var).to(gpu) * f2 * all_a_eff[i] @ (m_matrices[i] @ m_matrices[i].t() @ net.Ls[i].weight.detach().t() @ all_a_eff[i]).t()
        
        a_grad = f1 * forward_mat @ stimuli.x1.t() - f2 * args.noise_var * forward_mat @ forward_mat.t() @ a

    elif args.loss == 'MSE':
        coef = (stimuli.x1 @ v_eff - stimuli.x1 @ stimuli.x1_normed.t()).to(gpu)
        for i in range(args.n_layers):
            W_grads[i] = all_a_eff[i] @ (coef * m_matrices[i] @ stimuli.x1.t() + (torch.ones(1) * stimuli.noise_var).to(gpu) * m_matrices[i] @ (v_eff - stimuli.x1_normed.t())).t()
        
        a_grad = delta**2 * (1 - a.t() @ forward_mat @ stimuli.x1_normed.t()) * forward_mat @ stimuli.x1_normed.t() + args.noise_var * forward_mat @ stimuli.x1_normed.t() - args.noise_var * forward_mat @ forward_mat.t() @ a
        
        # get gradient for a
        

    # apply the phi_prime factors to the gradients
    for phi_prime_mask, grad in zip(repeated_phi_prime, W_grads):
        grad *= phi_prime_mask


    # remember the update is proportional to the negative gradient

    # print(f1)
    if iteration % args.test_interval == 0:
        all_delW, _a, _all_phi_prime, _v_eff = train_utils.create_variables_to_store(stimuli, net, args)

        rm_fields = -stimuli.x1 @ v_eff + torch.normal(mean=torch.zeros(10000), std=np.sqrt(stimuli.noise_var)).to(gpu) * torch.norm(v_eff)

        if args.loss == 'BCE':
            te_loss = torch.mean(-torch.log(1 - torch.sigmoid(rm_fields)))
        elif args.loss == 'MSE':
            noise_instant = torch.normal(mean=torch.zeros(50000), std=np.sqrt(stimuli.noise_var)).to(gpu)
            rm_target = -stimuli.x1 @ stimuli.x1_normed.t() + noise_instant
            rm_fields_test = -stimuli.x1 @ v_eff + noise_instant * torch.norm(v_eff)

            te_loss = torch.mean((rm_target -  rm_fields_test)**2)

        print(te_loss)
        te_error = torch.mean((rm_fields > 0).float())
        trainer.report_and_track(list_of_keys=['delW', 'a', 'phi_prime', 'v_eff'],
                                 list_of_vars=[all_delW, _a, _all_phi_prime, _v_eff],
                                 step=iteration,
                                 test_error=float(te_error.cpu()),
                                 test_loss=float(te_loss.cpu()))  # also saves data from each test into the hard drive

        # if trainer.detect_end_of_training(detect_mld=False):
        #     break

    # do gradient descent for the weights
    for W_grad, L in zip(W_grads, net.Ls):
        L.weight.data -= args.eta * W_grad
    
    # apply L2 constraint
    for i in range(len(net.Ls)):
        net.Ls[i].weight.data -= args.eta * args.lambda2 * (net.Ls[i].weight.data - init_weights[i])
    
    # do gradient descent for the readout
    a -= args.eta * a_grad

#%%
changes_to_weights = [_L.weight.data - init_w for _L, init_w in zip(net.Ls, init_weights)]

# print([np.linalg.norm(mat) for mat in changes_to_weights])


"""Test transfer with relearnd readouts"""
errors_with_new_readouts = train_utils.test_at_angles_meanfield(net, args, device=gpu, npoints=32)[1]
# errors_with_new_readouts = train_utils.test_at_angles(net, args, npoints=16)[1]


"""Test transfer with original readouts (no need to reload readout because it's not affected by RESET_WEIGHTS)"""
errors_with_original_readouts = train_utils.test_transfer_with_original_readouts(net, changes_to_weights, stimuli)

args._duration = time.time() - trainer.tic

mgr.update_data('args', args)
mgr.update_data('errors_with_original_readouts', errors_with_original_readouts)
mgr.update_data('errors_with_new_readouts', errors_with_new_readouts)

"""Save data. If running on cluster, the interpreter is ended after this cell."""

mgr.write('All simulations are completed.')

if args.cluster:
    quit()


# %% TEMP
plt.figure()
plt.plot(errors_with_new_readouts)
# plt.plot(_v_eff)
