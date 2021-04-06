#%%
import numpy as np
import torch, pickle, time, os, utils, model, train_utils
import matplotlib.pyplot as plt
import theory_utils as t_utils
import response_utils as r_utils

parser = train_utils.Args('1D Gabor')
parser.add('nonlinearity', 'relu')
parser.add('loss', 'MSE') # MSE or BCE

# model parameters
parser.add('N', 500); parser.add('Nhid', 500); parser.add('n_layers', 3)

# task parameters (make sure these values are floats)
parser.add('sig_w', 0.4); parser.add('sig_s', 0.2); parser.add('theta', np.pi)

# training parameter
parser.add('n_train_trials', 500)
parser.add('n_test_trials', 2500)  # remember we do 20 copies of this
parser.add('noise_var', 1.0)

args = parser.parse_args()

"""Set up GPUs. To run locally / on CPUs, set gpu=None."""

a_norms = []
sig_w_arr = np.linspace(0.1, 1, 30)

for i in range(30):
    args.sig_w = sig_w_arr[i]

    gpu = None
    stimuli = train_utils.GaborStimuli(args, device=gpu, simple_mode=True, verbose=False)
    x1 = stimuli.x1_normed.t().cpu().numpy()

    net = model.Model(args, dims=[args.N, *[args.Nhid for i in range(args.n_layers)]])

    W_effs = r_utils.get_effective_weights(net, stimuli.x0)[0]

    if args.n_layers == 3:
        forward_mat = W_effs[2] @ W_effs[1] @ W_effs[0]
    elif args.n_layers == 2:
        forward_mat = W_effs[1] @ W_effs[0]
    elif args.n_layers == 1:
        forward_mat = W_effs[0]

    _u, _s, _v = np.linalg.svd(forward_mat)
    xvsq = np.linalg.norm(_v @ x1)**2
    delta = np.linalg.norm(stimuli.x1.cpu().numpy())
    coef = (1 + delta**2 / stimuli.noise_var) / (1 + delta**2 / stimuli.noise_var * xvsq)

    noise_covar = forward_mat @ forward_mat.T
    _u, _s, _v = np.linalg.svd(noise_covar)
    a = np.linalg.inv(noise_covar + np.eye(noise_covar.shape[0]) * 1e-5) @ forward_mat @ x1
    a_real = a * coef

    a_norms.append(np.linalg.norm(a_real))

plt.figure()
plt.plot(sig_w_arr, a_norms)
#%%
plt.plot(np.log10(_s), marker='o')