import numpy as np
import torch
import matplotlib.pyplot as plt
"""
Created June 2019

Last audited: 08/17/19

= = = = = = = = = =
Updated 11/18/2019:
Include support for different hidden layer widths.
Allow selection of which layers to regularize.

Update 06/25/2020"
Removed lateral inhibition.
= = = = = = = = = =
"""

class Model(torch.nn.Module):

    def __init__(self, args, dims=None, verbose=True):

        super(Model, self).__init__()

        # read parameters
        self.N = args.N  # input dimensions
        self.n_layers = args.n_layers

        if dims is None:
            dims = [args.N, *[args.Nhid for i in range(args.n_layers)]]
        try:
            self.freeze = args.freeze
        except:
            self.freeze = []
        try:
            self.filter_size = args.filter_size
        except:
            self.filter_size = 1

        try:
            self.n_filters = args.n_filters
        except:
            self.n_filters = [1, *[1 for i in range(args.n_layers)]]

        try:
            self.penalty_mode = args.penalty_mode
        except:
            self.penalty_mode = 'less_RO'

        self.dims = dims; self.sig_w = args.sig_w

        # check validity of some parameters
        assert self.penalty_mode in ['all', 'less_RO', 'less_last']
        assert len(self.dims) == self.n_layers + 1

        '''Generate the layers'''
        self.Ws = []; self.Ls = []
        for ind in range(self.n_layers):
            self.add_module('L' + str(ind),
                            torch.nn.Linear(self.dims[ind], self.dims[ind+1], bias=False))
            self.Ls.append(getattr(self, 'L' + str(ind)))
            self.Ws.append(None)

            # freeze selected layers
            if ind in self.freeze:
                for p in list(self.Ls[-1].parameters()):
                    p.requires_grad = False

        self.RESET_WEIGHTS()

        '''Create readout'''
        self.RO = torch.nn.Linear(self.dims[-1], 1, bias=False)

        self.ref_weights = None

        #control which weights are regularized
        self.regularized_params = []
        for ind in range(len(self.Ls)):
            if self.penalty_mode == 'less_last' and ind == len(self.Ls) - 1:
                continue
            self.regularized_params += list(self.Ls[ind].parameters())

        if self.penalty_mode == 'all':
            self.regularized_params += list(self.RO.parameters())


    def RESET_WEIGHTS(self):
        device = self.L0.weight.device  # remember which device the model is on
        for ind in range(self.n_layers):
            W = make_multi_width_weights(self.n_filters[ind], self.n_filters[ind + 1], self.filter_size, self.sig_w, self.dims[ind], self.dims[ind+1])
            W -= W.mean()
            self.Ls[ind].weight.data = torch.from_numpy(W).float().t()
            self.Ws[ind] = self.Ls[ind].weight.detach().numpy()

        self.init_weights = [param.detach().clone() for param in list(self.parameters())]
        self.to(device)

    def get_obs(self, x, to_layer=None, threshold=0):
        n_trials = x.shape[0]
        if to_layer is None:
            to_layer = self.n_layers

        for ind in range(to_layer):
            x = self.Ls[ind](x).view(n_trials, self.n_filters[ind + 1], self.dims[ind + 1])
            threshold_mat = torch.ones_like(x).to(x.device) * threshold
            x = torch.relu(x - threshold_mat).reshape(n_trials, -1)
        return x

    def forward(self, x):
        return self.RO(self.get_obs(x))

    # def set_ref(self):
    #     # first set readout weight to zero such that the reference is zero
    #     self.RO.weight.data.fill_(0)
    #     self.RO.bias.data.fill_(0)
    #     self.ref_weights = [p.data.clone() for p in self.regularized_params]


def make_multi_width_weights(n_in_channels, n_out_channels, filter_size,
                      sig_w, in_neurons, out_neurons):


    block_list = []
    for _out_ind in range(n_out_channels):

        if _out_ind == 0:
            diff_angle = np.linspace(0, 2*np.pi, in_neurons, endpoint=False).reshape((in_neurons, 1)) -\
                            np.linspace(0, 2*np.pi, out_neurons, endpoint=False).reshape((1, out_neurons))

            # diff_angle[diff_angle < -np.pi] += np.pi * 2
            # diff_angle[diff_angle > np.pi] -= np.pi * 2
            # w = np.exp((-diff_angle**2 / 2 / sig_w**2))
            w = np.exp((np.cos(diff_angle) - 1) / sig_w**2)
            # w = np.exp((np.cos(diff_angle) - 1) / sig_w**2) * (-np.sin(diff_angle))

            w /= np.linalg.norm(w[:, 0]) * np.sqrt(in_neurons)
            blocks = w
        else:
            raise NotImplementedError('Multi widths with multiple filters not implemented.')

        block_list.append(blocks)

    return blocks

def bias_fig(model, ylim=None):
    fig, axes = plt.subplots(nrows=model.n_layers)
    if model.n_layers == 1:
        plt.plot(model.L0.bias.detach().numpy().flatten())
        plt.axhline(0, color='gray', ls='--')
        plt.title('L0 bias')
    else:
        axes = axes.flatten()
        for ind, L in enumerate(model.Ls):
            axes[ind].plot(L.bias.detach().numpy().flatten())
            axes[ind].axhline(0, color='gray', ls='--')
            axes[ind].set_title(f'L{ind} bias')
            if ylim is not None:
                axes[ind].set_ylim(-ylim, ylim)
    plt.tight_layout()


def activity_fig(model, gabortrainer):
    act_fig, axes = plt.subplots(nrows=model.n_layers)
    if model.n_layers == 1:
        axes = list([axes])
    else:
        axes = axes.ravel()

    #only use input for theta-dtheta
    batch = gabortrainer._add_noise(gabortrainer.test_mean[:gabortrainer.n_test_trials])

    for ind in range(model.n_layers):
        obs = model.get_obs(batch, ind + 1)
        axes[ind].errorbar(x=np.arange(obs.shape[1]), y=obs.detach().numpy().mean(axis=0),
             yerr=obs.detach().numpy().std(axis=0))
        axes[ind].set_title(f' X{ind + 1} activity')
    plt.tight_layout()

def pull_filters(conv_weight_mat, filter_size, N):
    n_filter_in = int(conv_weight_mat.shape[1] / N)
    n_filter_out = int(conv_weight_mat.shape[0] / N)
    mat = conv_weight_mat.detach().numpy()
    filters = np.zeros((n_filter_in, n_filter_out, N, filter_size))
    for i in range(n_filter_in):
        for j in range(n_filter_out):
            for k in range(N):
                row_ind = j * N + k
                col_ind = i * N + k
                if k <= N - filter_size:
                    filters[i, j, k, :] = mat[row_ind, col_ind:(col_ind + filter_size)]
                else:
                    filters[i, j, k, :(N - k)] =\
                    mat[row_ind, col_ind:((i+1)*N)]
                    filters[i, j, k, (N - k):] =\
                    mat[row_ind, i*N:((i*N+filter_size-(N-k)))]
    return filters


def get_filters(model, L, mode):
    assert L <= model.n_layers - 1
    assert mode in ['new', 'old', 'del']
    if mode == 'new':
        mat = model.Ls[L].weight
    if mode == 'old':
        mat = model.ref_weights[2*L]
    if mode == 'del':
        mat= model.Ls[L].weight - model.ref_weights[2*L]
    filters = pull_filters(mat, model.filter_size, model.N)
    return filters


def w_fig(model:Model, mode='post'):
    assert mode in ['post', 'del']
    w_fig, axes = plt.subplots(nrows=1, ncols=model.n_layers, dpi=100, figsize=(model.n_layers*3.8, 3))
    if model.n_layers == 1:
        axes = [axes]
    else:
        axes = axes.ravel()

    for ind, W in enumerate(model.Ws):

        if mode == 'post':
            im = axes[ind].imshow(W)
        elif mode == 'del':
            lim = np.max(np.abs(W - model.ref_weights[ind].numpy()))
            im = axes[ind].imshow(W - model.ref_weights[ind].numpy(), cmap='coolwarm', vmin=-lim, vmax=lim)
        w_fig.colorbar(im, ax=axes[ind])
        axes[ind].set_title(f'W{ind+1}, |DelW|={np.linalg.norm(W - model.init_weights[ind].numpy()):.3f}')
    plt.tight_layout()
