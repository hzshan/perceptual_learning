#%%
import numpy as np
import torch, pickle, time, os, utils, model, utils
import matplotlib.pyplot as plt

# %load_ext autoreload
# %autoreload 2

"""
Created Mar 14 2020
Last inspected: SEPT 4 2020
Last updated Aug 16 2020
Notes:
1. Can run model with any number of layers.
2. This also carefully reshapes all the variables stored such that all vectors are column vectors.
3. ALWAYS using ReLU nonlinearity. tanh support may be added later.
4. Writes to the result file at every test.
5. Memory-saving averaging -- reduces the test batch size, and do averages over small batches. This significantly reduces VRAM needed.
"""

parser = utils.Args('1D Gabor')
parser.add('nonlinearity', 'relu')
parser.add('loss', 'MSE') # MSE or BCE

# model parameters
parser.add('N', 1000); parser.add('Nhid', 1000); parser.add('n_layers', 3)

# task parameters (make sure these values are floats)
parser.add('sig_w', 0.8); parser.add('sig_s', 0.2); parser.add('theta', np.pi)

# training parameter
parser.add('eta', .32)
parser.add('n_learn', 2000000)
parser.add('n_train_trials', 500)
parser.add('n_test_trials', 2500)  # remember we do 20 copies of this
parser.add('test_interval', 500)
parser.add('noise_var', 0.01)
parser.add('lambda2', 0.0)  # Aug 16, 2020: added L2 constraint support

args = parser.parse_args()

"""Set up GPUs. To run locally / on CPUs, set gpu=None."""
gpu = None
if args.cluster == 1:
    gpu = torch.device('cuda:0')


TIC = time.time()  # Start time of the script

mgr = utils.Manager(args, device=gpu); trainer = utils.Trainer(args); stimuli = utils.GaborStimuli(args, device=gpu)

net = model.Model(args, dims=[args.N, *[args.Nhid for i in range(args.n_layers)]])
net = net.to(device=gpu)

#%%

utils.pretrain_readout(network=net, manager=mgr, stimuli=stimuli, loss_type=args.loss)

args.eta = float(args.eta / torch.norm(net.RO.weight).cpu())
"""Full training"""

# turn gradient back on
net.RESET_WEIGHTS()
net = net.to(device=gpu)

for i in range(net.n_layers):
    if i in net.freeze:
        continue
    else:
        net.Ls[i].weight.requires_grad = True

trainer.start_trial(net, stimuli, args, mgr)  # also logs args and time into the data dict in mgr

init_weights = [w.to(gpu) for w in net.init_weights]

all_init_vars = [_p.data.clone() for _p in list(net.parameters())]

for step in range(args.n_learn):

    # compute gradients using the chosen loss function
    utils.zero_and_grad(net, stimuli, loss_type=args.loss, v_teacher=stimuli.x1_normed.T)

    if step % args.test_interval == 0:

        te_error = 0; te_loss = 0
        for mem_saving_avg in range(20):
            _te_error, _te_loss, divergence_flag = utils.test(net, stimuli, loss_type=args.loss, v_teacher=stimuli.x1_normed)
            te_error += _te_error / 20
            te_loss += _te_loss / 20

            if divergence_flag:
                raise RuntimeError('Loss function diverged.')

        #create variables to store

        all_delW, a, all_phi_prime, v_eff = utils.create_variables_to_store(stimuli, net, args)

        trainer.report_and_track(list_of_keys=['delW', 'a', 'phi_prime', 'v_eff'],
                                 list_of_vars=[all_delW, a, all_phi_prime, v_eff],
                                 step=step,
                                 test_error=te_error,
                                 test_loss=te_loss)  # also saves data from each test into the hard drive

        if trainer.detect_end_of_training(detect_mld=False, threshold=1e-6):
            break

            
    utils.do_update(net, args.eta)
    
    
    # apply L2 constraint
    for p, p_init in zip(list(net.parameters()), all_init_vars):
        p.data -= args.eta * args.lambda2 * (p.data - p_init)


changes_to_weights = [_L.weight.data - init_w for _L, init_w in zip(net.Ls, init_weights)]

#%%
"""Test transfer with relearnd readouts"""
errors_with_new_readouts = utils.test_at_angles_meanfield(net, args, device=gpu, npoints=32)[1]

plt.figure()
plt.plot(utils.error_to_fi(errors_with_new_readouts))
#%%
"""Test transfer with original readouts (no need to reload readout because it's not affected by RESET_WEIGHTS)"""
errors_with_original_readouts = utils.test_transfer_with_original_readouts(net, changes_to_weights, stimuli)

args._duration = time.time() - trainer.tic

mgr.update_data('args', args)
mgr.update_data('errors_with_original_readouts', errors_with_original_readouts)
mgr.update_data('errors_with_new_readouts', errors_with_new_readouts)

mgr.write('All simulations are completed.')

if args.cluster:
    quit()
