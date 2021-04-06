import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import scipy.special
import torch, os, datetime, pickle, time, glob, model, warnings, argparse
import torch.optim, torch.nn
import scipy.interpolate as inter
from scipy.special import erfinv, erf
from scipy.optimize import curve_fit, fsolve
from sklearn.svm import LinearSVC

""" ======================================================================
================================ MATH  ==========================
========================================================================== """


def cos(vec_a, vec_b):
    a = np.array(vec_a); b = np.array(vec_b)
    return np.dot(a.flatten(), b.flatten()) / np.linalg.norm(a) / np.linalg.norm(b)


def error_to_fi(error):
    return (erfinv(error*2-1)*np.sqrt(2))**2

def fi_to_error(fi):
    return 1 - (erf(fi**0.5 / np.sqrt(2)) + 1) / 2

def get_corr_coefs(F):
    """
    Computes the noise correlations from F, the effective weight matrix
    """
    covar = F @ F.T
    norm_mat = np.linalg.norm(F, axis=1).reshape(-1, 1) @ np.linalg.norm(F, axis=1).reshape(1, -1) + 1e-10
    raw = covar / norm_mat
    raw -= np.eye(raw.shape[0])
    return (covar / norm_mat).flatten()

def matrix_prod(list_of_matrices, num_mats=None):
    '''
    Multiplies a list of matrices [W1, W2, W3,...,WN] as WN @ ... @ W3 @ W2 @ W1. 
    num_mats: number of matrices to be multiplied. 
    '''
    if num_mats is None:
        num_mats = len(list_of_matrices)
    output = list_of_matrices[0]
    for i in range(1, num_mats):
        output = list_of_matrices[i] @ output
    return output

def covar2corr(covar_matrix):
    diag = np.diag(covar_matrix).reshape(-1, 1)
    return covar_matrix / (np.sqrt(diag @ diag.T) + 1e-10) 

def normalize(vec):
    return vec / np.linalg.norm(vec)

""" ======================================================================
================================ SIMULATION  ==========================
========================================================================== """


def find_and_load_data(keyword, attribute=None):
    _name_list = np.sort(glob.glob("Run_Outcomes/*.data"))
    print('===================================================')
    counter = 0
    list_of_datasets = []
    for i in zip(range(len(_name_list)), _name_list):
        if keyword not in i[1]:
            continue
        print('#' + str(counter), ' '*(4-len(str(i[0]))), i[1][13:-5])
        counter += 1
        list_of_datasets.append(i[1])
    print('===================================================')
    if len(list_of_datasets) == 0:
        raise RuntimeError('No .data object containing this string is found.')
    
    if len(list_of_datasets) == 1:
        index = 0
    else:
        index = int(input('Enter index of the .data object to load.'))

    if attribute is None:
        text = ''
    else:
        text = attribute

    dataset = pickle.load(open(list_of_datasets[index], 'rb'))
    print(f'\n Loaded dataset: *** {list_of_datasets[index][13:-5]} ***, \n created {dataset[0]["_time"]} \n')
    print('{:<5}|{:<5}|{:<5}|{:<7}|{:<7}|{:<7}|{:<5}'.format('#', 'N', 'L', 'error', 'DelW1', "#entries", text))
    for i, entry in enumerate(dataset):
        if attribute is None:
            value = ''
        else:
            value = vars(entry['args'])[attribute]
        print('{:<5}|{:<5}|{:<5}|{:.4f} |{:.4f} |{:<7}|{:<5}'.format(i, entry["args"].N, entry["args"].n_layers, entry["error"][-1], np.linalg.norm(entry["delW"][0][-1]["s"]), len(entry["delW"][0]), value))
    
    return dataset



def create_variables_to_store(stimuli, network, args):

    with torch.no_grad():

        # created decomposed matrices to be stored
        all_delW = {}
        for i in range(args.n_layers):
            all_delW['layer_' + str(i)] = decompose_matrix(network.Ls[i].weight.detach().cpu().numpy().copy() - network.init_weights[i].numpy().copy(), ranks=3)
        a = network.RO.weight.detach().cpu().numpy().copy().T

        # computing the phi_primes with memory-saving averages (25 is a hyper parameter that means average over 25 batches)
        all_obs = [network.get_obs(stimuli.x0, to_layer=i+1) for i in range(network.n_layers)]

        all_phi_prime = []
        for obs in all_obs:
            obs[obs < 0] = 0
            obs[obs > 0] = 1
            obs = obs.cpu().numpy().copy().T
            all_phi_prime.append(obs)

        # convert all_phi_prime to numpy arrays on the cpu
        # compute effective readout
        W_tildes = []
        for i, _L in enumerate(network.Ls):
            W_tildes.append(all_phi_prime[i].reshape(-1, 1).repeat(_L.weight.shape[1], 1) * _L.weight.detach().cpu().numpy().copy())

        v_eff = a
        for w_tilde in W_tildes[::-1]:
            v_eff = w_tilde.T @ v_eff

    return all_delW, a, all_phi_prime, v_eff


def load_GD_changes(network:model.Model, data, time_index=-1, skip:list=[], readout=False):
    """Load changes to weights into the network object (will not load changes in layer numbers indiced by skip (0 index))"""
    network.RESET_WEIGHTS()
    for i in range(network.n_layers):
        if i in skip:
            continue
        delw = recover_matrix(data['delW'][i][time_index])
        network.Ls[i].weight.data += torch.from_numpy(delw).float().to(network.Ls[i].weight.device)

    if readout:
        network.RO.weight.data = torch.from_numpy(data['a'][-1]).float().to(network.RO.weight.device).view(1, -1)


def process_raw_data(name, skip_fails=False):
    source_dir = os.getcwd() + '/Raw_results/' + name + '/'
    target_dir = os.getcwd() + '/Run_Outcomes/' + name + '.data'

    """Load individual result files and append them to the same list"""
    raw_list = []

    for file_name in os.listdir(source_dir):
        if file_name[-7:] == 'results':
            if skip_fails:
                try:
                    entry = pickle.load(open(source_dir + file_name, 'rb'))
                    raw_list.append(entry)
                except:
                    print('Loading ' + source_dir + file_name + ' failed.')
            else:
                entry = pickle.load(open(source_dir + file_name, 'rb'))
                raw_list.append(entry)
    data = {}

    print(f'Number of entries: {len(raw_list)}')


    output_list = []
    for result_entry in raw_list:

        n_records = len(result_entry['loss'])

        v_eff = np.array(result_entry['v_eff']).squeeze()

        # compute order parameters in function space
        stimuli = GaborStimuli(result_entry['args'], simple_mode=True, verbose=False)

        x1 = stimuli.x1_normed.numpy()
        theta1 = (v_eff @ x1.T).squeeze()
        v_perp_raw_arr = v_eff - theta1.reshape(-1, 1) @ x1

        theta2 = np.linalg.norm(v_perp_raw_arr, axis=1)
        v_perp_arr = v_perp_raw_arr / theta2.reshape(-1, 1)

        # Re organize the phi_prime datas. Get a list for phi_prime for each layer with length equal to n_records
        phi_prime = [[] for i in range(result_entry['args'].n_layers)]

        for i in range(result_entry['args'].n_layers):
            for t in range(n_records):
                phi_prime[i].append(result_entry['phi_prime'][t][i])

        phi_prime = [np.array(phi_prime_list) for phi_prime_list in phi_prime]  # convert each new list into a numpy array of dimension (n_records, n_dim)

        # reorganize the changes to weights similarly (create n_layers lists, each of length n_records )
        delW = [[] for i in range(result_entry['args'].n_layers)]
        for i in range(result_entry['args'].n_layers):
            for t in range(n_records):
                delW[i].append(result_entry['delW'][t]['layer_' + str(i)])

        output_list_entry = {'loss':result_entry['loss'], 'error': result_entry['error'], 'a':np.array(result_entry['a']).squeeze(), 'v_eff':v_eff, 'x1':x1, 'theta1':theta1, 'theta2':theta2, 'phi_prime':phi_prime, 'delW':delW}
        output_list_entry['args'] = result_entry['args']; output_list_entry['_time'] = result_entry['time']

        # if including transfer results, add those as well
        list_of_keys_to_try = ['errors_with_original_readouts', 'errors_with_new_readouts', 'pre_training_subsampled_error',
                              'post_training_subsampled_error', 'sigs_error']
        for key in list_of_keys_to_try:
            
            if key in result_entry.keys():
                output_list_entry[key] = result_entry[key]

        output_list.append(output_list_entry)

    """Save processed data"""
    pickle.dump(output_list, open(target_dir, 'wb'))

""" ======================================================================
================================ SGD  ==========================
========================================================================== """

def check_convergence(sequence, patience:int, tolerance=0):
    """
    Check whether a certain metric (error, loss etc. ) has converged. Returns a Boolean. Assuming that the metric should be decreasing.
    """
    if len(sequence) < 2:
        return False
    else:
        # count the number of times where the metric at time t is higher than that at t-1
        violations = np.sum((np.array(sequence) - np.roll(np.array(sequence), 1)) > tolerance) - 1  # minus 1 because the latest metric is moved to the first
        if violations > patience:
            return True
        else:
            return False


def zero_and_grad(model, stimuli, loss_type, v_teacher=None, mode='train'):
    assert mode in ['train', 'test']
    assert loss_type in ['BCE', 'MSE']
    if loss_type =='MSE':
        assert v_teacher is not None
        v_teacher = v_teacher.view(-1, 1)

    if loss_type == 'BCE':
        if mode == 'train':
            r_arr = model(stimuli.tr_stim())
            loss = torch.mean(-stimuli.train_labels*torch.log(1e-5 + torch.sigmoid(r_arr))-\
                                   (1-stimuli.train_labels)*torch.log(1e-5 + 1-torch.sigmoid(r_arr))).view(-1)
        elif mode == 'test':
            r_arr = model(stimuli.te_stim())
            loss = torch.mean(-stimuli.test_labels*torch.log(1e-5 + torch.sigmoid(r_arr))-\
                                   (1-stimuli.test_labels)*torch.log(1e-5 + 1-torch.sigmoid(r_arr))).view(-1)
    elif loss_type == 'MSE':
        if mode == 'train':
            _stim = stimuli.tr_stim()
            r_arr = model(_stim); r_teach = _stim @ v_teacher
            loss = torch.mean((r_arr.view(-1) - r_teach.view(-1))**2)
        elif mode == 'test':
            _stim = stimuli.te_stim()
            r_arr = model(_stim); r_teach = _stim @ v_teacher
            loss = torch.mean((r_arr.view(-1) - r_teach.view(-1))**2)
    model.zero_grad()
    loss.backward()

def test(model, stimuli, loss_type, v_teacher=None):

    assert loss_type in ['BCE', 'MSE']
    if loss_type == 'MSE':
        assert v_teacher is not None
        v_teacher = v_teacher.view(-1, 1)

    divergence_flag = False

    _stim = stimuli.te_stim()
    with torch.no_grad():
        test_r_arr = model(_stim)
        err = torch.mean((test_r_arr * (2 * stimuli.test_labels-1) <= 0).float())

        if loss_type == 'BCE':
            test_loss = torch.mean(-stimuli.test_labels * torch.log(1e-5 + torch.sigmoid(test_r_arr))-\
                               (1-stimuli.test_labels) * torch.log(1e-5 + 1-torch.sigmoid(test_r_arr)))

        elif loss_type == 'MSE':
            r_teach = _stim @ v_teacher
            test_loss = torch.mean((test_r_arr.view(-1) - r_teach.view(-1))**2)

        if torch.isnan(test_loss):
            warnings.warn('Loss function diverged.')
            divergence_flag = True
    return float(err.detach()), float(test_loss.detach()), divergence_flag

def do_update(model, eta):
    for p in list(model.parameters()):
        if p.grad is None:
            continue
        else:
            p.data -= p.grad.detach() * eta

def svm_readout(model, stimuli):
    bl_error, v_opt, b_opt = test_linear(model, stimuli, return_weights=True)
    norm_factor = np.linalg.norm(v_opt)
    model.RO.weight.data = torch.from_numpy(v_opt).float() / norm_factor
    model.RO.bias.data = torch.ones(1) * float(b_opt) / norm_factor


class Trainer:
    """
    Built 09/25/19
    Audited 09/26/19
    Last updated: 03/14/2020
    """
    def __init__(self, args):
        try:
            self.stepwise = args.stepwise
        except:
            self.stepwise = 0
        try:
            self.gamma = args.gamma
        except:
            self.gamma = 0

        self.trial_data = {'loss':[], 'error':[]}

        self.reached_MLD = False  # flag showing whether model reached MLD

    def start_trial(self, model, stimuli, args, manager):
        self.model = model
        self.stimuli = stimuli
        self.args = args
        self.tic = time.time()
        self.manager = manager

        self.manager.trial_data_dict['time'] = str(datetime.datetime.now())
        self.manager.trial_data_dict['args'] = args

    def _calc_regularization(self, model, args, ewc_coefs=None):

        try:
            lambda1 = args.lambda1
        except:
            lambda1 = 0

        try:
            lambda2 = args.lambda2
        except:
            lambda2 = 0

        try:
            lambda_ewc = args.lambda_ewc
        except:
            lambda_ewc = 0

        reg_loss = torch.zeros(1)
        assert len(model.regularized_params) == len(model.ref_weights)

        # calculate EWC penalty
        if lambda_ewc > 0:
            assert ewc_coefs is not None
            assert len(ewc_coefs) == len(model.regularized_params)
            for p_ind, param in enumerate(model.regularized_params):
                reg_loss += torch.sum((model.ref_weights[p_ind] - param)**2 * ewc_coefs[p_ind]) * lambda_ewc

        # calculate L1 penalty
        if lambda1 > 0:
            for p_ind, param in enumerate(model.regularized_params):
                reg_loss += torch.norm(param - model.ref_weights[p_ind], p=1) * lambda1

        if lambda2 > 0:
            for p_ind, param in enumerate(model.regularized_params):
                reg_loss += torch.norm(param - model.ref_weights[p_ind], p=2)**2 * lambda2

        return reg_loss

    def _supralinear_update(self, model, args):
        for p in list(model.parameters()):
            mask = torch.abs(p.grad.detach())**args.gamma
            mask /= torch.max(mask)
            update = mask * p.grad.detach()
            p.data -= update * args.eta

    def _truncate_gradients(self, model, prev_weights):
        for p_ind, param in enumerate(model.regularized_params):
            _inds = (param.detach() - model.ref_weights[p_ind]) *\
             (prev_weights[p_ind] - model.ref_weights[p_ind]) < 0
            param.data[_inds] = model.ref_weights[p_ind][_inds]

        # self.writer.write('Trainer: network readout has been optimized before full training.')

    def detect_end_of_training(self, detect_mld=True, threshold=1e-10):
        """
        Returns a boolean value indicating whether end-of-training criteria have been met.
        detect_mld: if true, stop training when test error is smaller than MLD error; otherwise, stop at convergence of loss function
        """

        end_flag = False
        if detect_mld:
            if self.trial_data['error'][-1] <= self.stimuli.mld_err:
                self.manager.write('\n GaborTrainer: Training ended because error reached MLD error.')
                self.reached_MLD = True
                self.exit_code = 0
                end_flag = True

        else:
            if self.args.loss == 'MSE':
                if self.trial_data['loss'][-1] < threshold:
                    self.manager.write('\n GaborTrainer: Training ended because loss is less than 1e-6.')
                    self.exit_code = 1
                    end_flag = True
            elif self.args.loss == 'BCE':
                if len(self.trial_data['loss']) > 20:
                    if self.trial_data['loss'][-1] > np.mean(self.trial_data['loss'][-20:]):
                        self.manager.write('\n GaborTrainer: Training ended because loss function converged.')
                        self.exit_code = 1
                        end_flag = True
        return end_flag


    def report_and_track(self, list_of_keys:list, list_of_vars:list, step, test_error, test_loss):
        """
        record test_loss, test_error. Write to the report.
        list_of_keys: list of keys to use in the trial_data dictionary
        list_of_vars: correspond to the list of keys
        """
        new_data_dict = {}
        new_data_dict['error'] = test_error
        new_data_dict['loss'] = test_loss
        self.trial_data['loss'].append(float(test_loss))
        self.trial_data['error'].append(float(test_error))

        assert len(list_of_keys) == len(list_of_vars)
        for var, key in zip(list_of_vars, list_of_keys):
            new_data_dict[key] = var

        msg = f'{step + 1}/{self.args.n_learn}, Error {test_error:.5f}, Training Loss {float(test_loss):.5f}, Time {int(time.time() - self.tic)} sec'
        if self.manager is None:
            print(msg)
        else:
            self.manager.write(msg)

        # log the new data into the running list of entries
        self.manager.append_data_and_save(new_data_dict)


class Manager:
    """
    Created Mar 15 2020
    Stores results from each trial in a dictionary object in a batch-specific folder
    Adds data during training (rather than after training is completed)
    """
    def __init__(self, args, device=None, suffix=''):

        self.on_cluster = args.cluster
        self.device = device
        self.n_entries = 0 # tracks how many times the append_data_and_save method has been called.

        self.result_dir = os.getcwd() + f'/Raw_results/{args.BATCH_NAME}' + suffix
        self.report_name = self.result_dir + f'/Report_{args.TRIAL_IND}.txt'

        self.id = args.TRIAL_IND

        self.trial_data_dict = {}

        if self.on_cluster == 1:
            mode_name = 'cluster'

            # Create report file
            self.report = open(self.report_name, 'w')
            self.report.close()

            # If the result directory does not exist, create one (sometimes apparently the directory is created by another trial, then wait a little)
            if os.path.isdir(self.result_dir) is False:
                try:
                    os.mkdir(self.result_dir)
                except:
                    print('Failed to create the directory. It probably already exists. Proceeding.')

            self.write(str(args))

        else:
            mode_name = 'local'

        self.write(f'Running in {mode_name} mode. Run started at ' + str(datetime.datetime.now()))

    def append_data_and_save(self, new_data:dict):
        """
        The new data dictionary should contain input for only the latest point in time
        The manager then checks whether this is the first entry in time. If so, create a list for each key.
        If this is not the first entry, append the variable from each key to the corresponding list.
        Updated Mar 28 2020: To save memory, the self_trial_data_dict object will be set to None after each call to this function.
        """
        if self.n_entries > 0 and self.on_cluster==1:
            self.trial_data_dict = pickle.load(open(self.result_dir + '/' + f'{self.id}.results', 'rb'))

        self.n_entries += 1
        for key in new_data.keys():
            if key in self.trial_data_dict.keys():
                self.trial_data_dict[key].append(new_data[key])
            else:
                self.trial_data_dict[key] = [new_data[key]]

        # always overwrites the existing file
        if self.on_cluster == 1:
            pickle.dump(self.trial_data_dict, open(self.result_dir + '/' + f'{self.id}.results', 'wb'))
        else:
            print('Data not added since running in local mode.')

        # release memory
        if self.on_cluster == 1:
            self.trial_data_dict = None

    def write(self, text, new_line=True):
        print(text)
        if self.on_cluster == 1:
            file = open(self.report_name, 'a')
            if new_line:
                file.write('\n' + text)
            else:
                file.write(' | ' + text)
            file.close()


    def update_data(self, key, data):
        """
        Temporarily load the data dictionary and only update the args. Useful for adding additional attributes.
        """
        if self.on_cluster == 1:
            data_dict = pickle.load(open(self.result_dir + '/' + f'{self.id}.results', 'rb'))
            data_dict[key] = data
            pickle.dump(data_dict, open(self.result_dir + '/' + f'{self.id}.results', 'wb'))

class Args:
    """
    Simple wrapper for argumentparser. Makes code cleaner
    """
    def __init__(self, description=None):
        self.p = argparse.ArgumentParser(description=description)
        self.p.add_argument('-f')
        print('Warning: argument type specified by the default value.')
        self.add('cluster', default=0)
        self.add('BATCH_NAME', default='BATCH_NAME')
        self.add('TRIAL_IND', default=0)

    def add(self, name, default, ptype=None, help=None, optional=True):
        name = name
        if optional:
            name = '--' + name
        if ptype is None:
            ptype = type(default)
        self.p.add_argument(name, type=ptype, default=default, help=help)

    def parse_args(self):
        args, unknown = self.p.parse_known_args()
        args.f = None
        return args


def pretrain_readout(network, manager, stimuli, loss_type, n_learn=500000, return_error=False,
    error_converge=False, force_gd=False):
    """
    error_converge: whether to detect convergence and end training based on the training error. Default False
    force_gd: by default, when using MSE loss the readout is given by a pseudo-inverse formula. Setting 
    force_gd to True will force using gradient descent
    epsilon: controls how matrix inversion is done when calculating optimal a. If epsilon is None, then only the first few ranks are inverted. Otherwise, an identity matrix times epsilon is added to the matrix and then inverted.
    """
    def _train(optim):
        pretrain_loss = []; pretrain_error = []
        for i in range(n_learn):
            zero_and_grad(network, stimuli, loss_type=loss_type, v_teacher=v)

            optim.step()

            if (i + 1) % 4000 == 0:

                te_error = 0; te_loss = 0
                for mem_saving_avg in range(20):
                    _te_error, _te_loss, divergence_flag = test(network, stimuli, loss_type=loss_type, v_teacher=v)
                    te_error += _te_error / 20
                    te_loss += _te_loss / 20

                if divergence_flag:
                    manager.write('Pretraining the readout with learning rate of 1 diverged. Reducing learning rate to 0.1. (if this msg appears a second time, the reduced learning rate version has already been run.')
                    break

                manager.write(f'{te_error:.3f}, {te_loss:.4f}, a norm: {torch.norm(network.RO.weight.detach()):.3f}, a_grad_norm: {torch.norm(network.RO.weight.grad.detach()):.3f}')
                pretrain_loss.append(te_loss)
                pretrain_error.append(te_error)

                # Check for convergence
                if error_converge:
                    if utils.check_convergence(pretrain_error, patience=5):
                        manager.write('Convergence threshold (for error) reached.')
                        break

                else:
                    if check_convergence(pretrain_loss, patience=3):
                        manager.write('Convergence threshold reached.')
                        break
        return divergence_flag, te_error
    # Teacher weights (if using MSE loss)

    tic = time.time()
    v = stimuli.x1_normed.view(-1, 1)

    # Use a closed form for a if using MSE loss (added June 2 2020)
    if loss_type == 'MSE' and force_gd is False:
        eff_weights = get_effective_weights(network, stimuli.x0, full_mat=True)[0]
        forward_mat = eff_weights[0]
        for i in range(len(eff_weights)):
            if i > 0:
                forward_mat = eff_weights[i] @ forward_mat
        
        '''Uses as many singular vectors as possible, as long as numerically stable'''
        if network.n_layers == 1:
            numpy_a = mse_optimal_a(forward_mat, stimuli, 3)
        else:
            numpy_a = mse_optimal_a(forward_mat, stimuli)
        network.RO.weight.data = torch.FloatTensor(numpy_a).view(1, -1).to(network.RO.weight.device)
        
        te_stim = stimuli.te_stim()
        target = te_stim @ v
        net_output = network(te_stim)
        te_error = float(torch.mean((net_output * stimuli.test_labels < 0).float()).cpu())
        manager.write('Readout is "pretrained" using the expression for optimal readout.')
    else:
        # If using BCE loss, or using MSE loss but forcing gradient descent, do gradient descent

        # initialize readout as Gaussian random
        network.RO.weight.data = torch.normal(torch.zeros_like(network.RO.weight), 1 / np.sqrt(network.RO.weight.shape[1])).to(network.RO.weight.device)

        # Turn gradients for weight matrices off
        for L in network.Ls:
            L.weight.requires_grad = False

        # Set a default learning rate of 1. If the loss diverges, then try again with .1.
        optim = torch.optim.SGD(list(network.RO.parameters()), lr=1)
        divergence_flag, te_error = _train(optim)

        if divergence_flag:
            # If training with lr=1 failed, reset the weights, create a new optim with lower learning rate, and try again
            network.RO.weight.data = torch.normal(torch.zeros_like(network.RO.weight), 1 / np.sqrt(network.RO.weight.shape[1])).to(network.RO.weight.device)
            optim = torch.optim.SGD(list(network.RO.parameters()), lr=.1)
            divergence_flag2, te_error = _train(optim)

            # Throw an AssertionError if training diverged again.
            assert divergence_flag2 is False

    # record norm of the initial readout
    manager.write(f'Norm of initial a: {torch.norm(network.RO.weight)}')
    manager.write(f'Pretraining took {(time.time() - tic):.1f} secs.')

    if return_error:
        return te_error


""" ======================================================================
================================ STIMULI  ==========================
========================================================================== """



class GaborStimuli:
    def __init__(self, args, verbose=True, device=None, simple_mode=False):
        """
        If device is None, it assumes that everything will be on the gpu. Otherwise, enter a string for the gpu e.g. "cuda"
        """
        self.device = device

        # updated Aug 7: delta will now be set such that norm of the signal is one

        delta_temp = 1e-6
        x0p_temp = make_x0(args.theta + delta_temp, args.sig_s, args.N)
        x0m_temp = make_x0(args.theta - delta_temp, args.sig_s, args.N)
        slope = (x0p_temp - x0m_temp) / 2 / delta_temp
        delta = 1 / np.linalg.norm(slope) / 10
        self.delta = delta
        self.noise_var = args.noise_var

        self.n_train_trials = args.n_train_trials
        self.n_test_trials = args.n_test_trials

        x0p = make_x0(args.theta + delta, args.sig_s, args.N).reshape((1, args.N))
        x0m = make_x0(args.theta - delta, args.sig_s, args.N).reshape((1, args.N))

        if simple_mode:
            self.simple_mode = True
            self.train_mean = None; self.test_mean = None; self.train_labels = None; self.test_labels = None
        else:
            self.simple_mode = False
            self.train_mean = torch.from_numpy(np.vstack((np.repeat(x0p, args.n_train_trials, axis=0),
                                    np.repeat(x0m, args.n_train_trials, axis=0)))).float()
            self.test_mean = torch.from_numpy(np.vstack((np.repeat(x0p, args.n_test_trials, axis=0),
                                   np.repeat(x0m, args.n_test_trials, axis=0)))).float()
            _tr_labels = torch.cat((torch.ones(args.n_train_trials), torch.zeros(args.n_train_trials)))
            _test_labels = torch.cat((torch.ones(args.n_test_trials),torch.zeros(args.n_test_trials)))
            self.train_labels = _tr_labels.view((2 * args.n_train_trials, 1))
            self.test_labels = _test_labels.view((2 * args.n_test_trials, 1))

        self.mld_err = calc_MLD((x0p - x0m) / 2, args.noise_var, verbose=verbose)
        self.x1_normed = ((x0p - x0m) / np.linalg.norm(x0p - x0m)).flatten()
        self.x0 = torch.from_numpy(x0p + x0m).float() / 2; self.x0_normed = self.x0 / torch.norm(self.x0)
        self.x1 = torch.from_numpy(x0p - x0m).float() / 2; self.x1_normed = self.x1 / torch.norm(self.x1)
        self.sig_s = args.sig_s

        if self.device is not None:

            self.train_labels = self.train_labels.to(device)
            self.test_labels = self.test_labels.to(device)
            self.x1_normed = self.x1_normed.to(device)
            self.x0_normed = self.x0_normed.to(device)
            self.x0 = self.x0.to(device)
            self.x1 = self.x1.to(device)
            self.train_mean = self.train_mean.to(device)
            self.test_mean = self.test_mean.to(device)

    def tr_stim(self):
        if self.simple_mode:
            raise RuntimeError('GaborStimuli created in simple mode. Therefore it cannot generate input.')
        if self.device is None:
            return torch.normal(self.train_mean, np.sqrt(self.noise_var))
        else:
            return self.train_mean + torch.cuda.FloatTensor(self.train_mean.shape).normal_(std=np.sqrt(self.noise_var))

    def te_stim(self):
        if self.simple_mode:
            raise RuntimeError('GaborStimuli created in simple mode. Therefore it cannot generate input.')
        if self.device is None:
            return torch.normal(self.test_mean, np.sqrt(self.noise_var))
        else:
            return self.test_mean + torch.cuda.FloatTensor(self.test_mean.shape).normal_(std=np.sqrt(self.noise_var))

class SigsStimuli:

    def __init__(self, args, verbose=True, device=None, simple_mode=False):
        """
        If device is None, it assumes that everything will be on the gpu. Otherwise, enter a string for the gpu e.g. "cuda"
        """
        self.device = device

        # updated Aug 7: delta will now be set such that norm of the signal is one

        delta_sigs_temp = 1e-6
        x0p_temp = make_x0(args.theta, args.sig_s + delta_sigs_temp, args.N)
        x0m_temp = make_x0(args.theta, args.sig_s - delta_sigs_temp, args.N)
        slope = (x0p_temp - x0m_temp) / 2 / delta_sigs_temp
        delta = 1 / np.linalg.norm(slope) / 10
        self.delta = delta
        self.noise_var = args.noise_var

        self.n_train_trials = args.n_train_trials
        self.n_test_trials = args.n_test_trials

        x0p = make_x0(args.theta, args.sig_s + delta, args.N).reshape((1, args.N))
        x0m = make_x0(args.theta, args.sig_s - delta, args.N).reshape((1, args.N))

        if simple_mode:
            self.simple_mode = True
            self.train_mean = None; self.test_mean = None; self.train_labels = None; self.test_labels = None
        else:
            self.simple_mode = False
            self.train_mean = torch.from_numpy(np.vstack((np.repeat(x0p, args.n_train_trials, axis=0),
                                    np.repeat(x0m, args.n_train_trials, axis=0)))).float()
            self.test_mean = torch.from_numpy(np.vstack((np.repeat(x0p, args.n_test_trials, axis=0),
                                   np.repeat(x0m, args.n_test_trials, axis=0)))).float()
            _tr_labels = torch.cat((torch.ones(args.n_train_trials), torch.zeros(args.n_train_trials)))
            _test_labels = torch.cat((torch.ones(args.n_test_trials),torch.zeros(args.n_test_trials)))
            self.train_labels = _tr_labels.view((2 * args.n_train_trials, 1))
            self.test_labels = _test_labels.view((2 * args.n_test_trials, 1))

        self.mld_err = calc_MLD((x0p - x0m) / 2, args.noise_var, verbose=verbose)
        self.x1_normed = ((x0p - x0m) / np.linalg.norm(x0p - x0m)).flatten()
        self.x0 = torch.from_numpy(x0p + x0m).float() / 2; self.x0_normed = self.x0 / torch.norm(self.x0)
        self.x1 = torch.from_numpy(x0p - x0m).float() / 2; self.x1_normed = self.x1 / torch.norm(self.x1)
        self.sig_s = args.sig_s

        if self.device is not None:

            self.train_labels = self.train_labels.to(device)
            self.test_labels = self.test_labels.to(device)
            self.x1_normed = self.x1_normed.to(device)
            self.x0_normed = self.x0_normed.to(device)
            self.x0 = self.x0.to(device)
            self.x1 = self.x1.to(device)
            self.train_mean = self.train_mean.to(device)
            self.test_mean = self.test_mean.to(device)

    def tr_stim(self):
        if self.simple_mode:
            raise RuntimeError('GaborStimuli created in simple mode. Therefore it cannot generate input.')
        if self.device is None:
            return torch.normal(self.train_mean, np.sqrt(self.noise_var))
        else:
            return self.train_mean + torch.cuda.FloatTensor(self.train_mean.shape).normal_(std=np.sqrt(self.noise_var))

    def te_stim(self):
        if self.simple_mode:
            raise RuntimeError('GaborStimuli created in simple mode. Therefore it cannot generate input.')
        if self.device is None:
            return torch.normal(self.test_mean, np.sqrt(self.noise_var))
        else:
            return self.test_mean + torch.cuda.FloatTensor(self.test_mean.shape).normal_(std=np.sqrt(self.noise_var))



def make_x0(theta, width, n):
    """
    Tuning curves of retinal cells
    """
    _pref = np.linspace(0, np.pi * 2, n, endpoint=False)
    _diff = _pref - theta
    #
    # _diff[_diff > np.pi] -= 2*np.pi
    # _diff[_diff < -np.pi] += 2*np.pi

    # _x0 = np.exp(-_diff**2 / width**2 / 2)
    _x0 = np.exp((np.cos(_diff) - 1) / width**2)
    _x0 /= np.linalg.norm(_x0) / np.sqrt(n)
    return _x0.reshape(n, 1)

def calc_MLD(dx, sig2, verbose=True):
    if sig2 == 0:
        if verbose:
            print('calc_MLD():NO NOISE. Error at optimum is zero.')
        ml_error = 0
        return ml_error
    else:
        _J0 = np.sum(dx**2) / sig2
        margin = np.sqrt(_J0) / np.sqrt(2)
        ml_error = float(0.5 - scipy.special.erf(margin) / 2)
        if verbose:
            print(f'calc_MLD(): MLD error is {ml_error}')

    return ml_error


""" ======================================================================
================================ MEAN FIELD THEORY  ==========================
========================================================================== """


def get_active_inds(network:model.Model, mean_input, threshold=0):
    """
    Get the indices of active neurons for all layers.
    """
    active_inds = [np.arange(network.N)]

    for l in range(network.n_layers):
        active_inds.append(np.where(network.get_obs(mean_input, to_layer=l+1, threshold=threshold).cpu().data.numpy().flatten() > 0)[0])
        assert len(active_inds[l] > 0)

    return active_inds

def get_effective_weights(network:model.Model, mean_input, full_mat=False, threshold=0):
    """
    Get weights for a linear network that is equivalent to the nonlinear network *for a specific stimulus*. It subtracts the mean from weight matrices.

    full_mat: if True, return matrices of the same dimension as original matrices
              if False, return matrices of dimensions #active neurons * #active neurons
    mean_input: 1-by-N torch tensor. should be stimuli.x0 
    Returns a linear of weight matrices (from first layer to last)
    """

    w_eff_list =[]
    active_inds = get_active_inds(network, mean_input, threshold=threshold)

    if full_mat:
        for l in range(network.n_layers):

            # make effective weight mask
            mean_obs = network.get_obs(mean_input, to_layer=l+1, threshold=threshold).cpu().data.numpy()
            mean_obs[mean_obs<0] = 0; mean_obs[mean_obs>0] = 1
            mask = np.repeat(mean_obs.reshape(-1, 1), network.Ls[l].weight.shape[1], axis=1)

            mat = network.Ls[l].weight.cpu().data.numpy().copy()

            w_eff_list.append(mask * mat)

    else:
        # Get indices of active neurons in each layer (in the input layer, all neurons are active)
        for i in range(network.n_layers):
            w_eff_list.append(network.Ls[i].weight.cpu().data.numpy()[active_inds[i+1]][:, active_inds[i]])
        
    return w_eff_list, active_inds



def J_from_effective_weights(list_of_weights, noise_var, x1):
    """
    Computes linear Fisher Info (J) with list of effective weights
    """
    n_layers = len(list_of_weights)
    if n_layers == 3:
        F = list_of_weights[2] @ list_of_weights[1] @ list_of_weights[0]
    elif n_layers == 2:
        F = list_of_weights[1] @ list_of_weights[0]
    elif n_layers == 1:
        F = list_of_weights[0]
    
    covar = F @ F.T * noise_var
    u, s, v = np.linalg.svd(covar)
    return float(x1.reshape(1, -1) @ F.T @ np.linalg.inv(covar + 1e-6 * s[0] * np.eye(covar.shape[0])) @ F @ x1.reshape(-1, 1))

def normalize_j(j, j0):
    return (j - j0) / np.max(j - j0)

""" ======================================================================
================================ MINIMUM PERTURBATION  ==========================
========================================================================== """

def predict_delw_twolayer(network:model.Model, stimuli, init_guess,
 k, update_coef=0.9, max_iter=500, convergence_thres=1e-7, Q_epsilon=0):
    """
    A wrapped version for two layers
    Output:
    order_params, delw1, delw2

    It solves the self-consistent equations for two-layer networks, and predict the resulting delw matrices.
    """
    x1 = stimuli.x1_normed.numpy().reshape(-1, 1)

    # Get bounds of the effective matrices
    w_effs, active_inds = get_effective_weights(network, stimuli.x0, threshold=0)
    W1, W2 = w_effs

    forward_mat = W2 @ W1

    a = mse_optimal_a(forward_mat, stimuli)
    a_norm_sq = np.linalg.norm(a)**2

    # Use the iterative solver to solve self-consistent equations
    if init_guess is not None:
        trajectories = [init_guess]
    else:
        trajectories = [[np.linalg.norm(W2.T @ a)**2, 0.1 / np.linalg.norm(W2.T @ a)**2]]

    for i in range(max_iter):

        u1, v1 = trajectories[-1]
        Q = (k - a_norm_sq * v1) * u1 * np.eye(W1.shape[1]) + a_norm_sq * W1.T @ W1
        Q += np.eye(Q.shape[0]) * Q_epsilon
        Y = (k - a_norm_sq * v1) * x1- k * W1.T @ W2.T @ a
        Lambda = np.linalg.inv(Q) @ Y
        u1_out = np.linalg.norm((k - a_norm_sq * v1)**-1 * (W2.T @ a * k + a_norm_sq * W1 @ Lambda))**2
        v1_out = np.linalg.norm(Lambda)**2

        trajectories.append((u1_out * (1-update_coef) + trajectories[i][0] * update_coef, v1_out * (1-update_coef) +\
             trajectories[i][1] * update_coef))

        if np.abs(u1_out - u1) / np.abs(u1) < convergence_thres and np.abs(v1_out - v1) / np.abs(v1) < convergence_thres:
            break

    delw1 = get_delw1(Lambda, trajectories[-1][1], a_norm_sq, a, W1, W2, k)
    delw2 = get_delw2(Lambda, trajectories[-1][1], a_norm_sq, a, W1, W2, k)

    return trajectories[-1], delw1, delw2, active_inds, a, trajectories


def predict_delw_twolayer_given_a(network:model.Model, stimuli, k, a):
    """
    Solve self-consistent equations for 2-Layer networks and give the delW expressions.
    k: parameter controlling relative strength of L2 constraint on the two matrices. k=1 is equal strength.
    a: readout to use. Must be a column vector of length Nhid
    Output:
    order_params, delw1, delw2, active_inds, a

    It solves the self-consistent equations for two-layer networks, and predict the resulting delw matrices.
    """
    x1 = stimuli.x1_normed.numpy().reshape(-1, 1)

    # Get bounds of the effective matrices
    w_effs, active_inds = get_effective_weights(network, stimuli.x0)
    W1, W2 = w_effs

    a_norm_sq = np.linalg.norm(a)**2


    # Use the iterative solver to solve self-consistent equations
    trajectories = [(100, 0.1 / a_norm_sq)] # the format should be (u1, v1)
    max_iter = 100

    for i in range(max_iter):

        u1, v1 = trajectories[-1]
        Q = (k - a_norm_sq * v1) * u1 * np.eye(W1.shape[1]) + a_norm_sq * W1.T @ W1
        Y = (k - a_norm_sq * v1) * x1- k * W1.T @ W2.T @ a
        Lambda = Lambda_def(Q, Y)
        u1_out = np.linalg.norm((k - a_norm_sq * v1)**-1 * (W2.T @ a * k + a_norm_sq * W1 @ Lambda))**2
        v1_out = np.linalg.norm(Lambda)**2

        trajectories.append((u1_out, v1_out))

        if np.abs(u1_out - u1) / np.abs(u1) < 1e-7 and np.abs(v1_out - v1) / np.abs(v1) < 1e-7:
            break



    delw1 = get_delw1(Lambda, trajectories[-1][1], a_norm_sq, a, W1, W2, k)
    delw2 = get_delw2(Lambda, trajectories[-1][1], a_norm_sq, a, W1, W2, k)

    return trajectories[-1], delw1, delw2, active_inds, a


def predict_delw_fsolve_twolayer(network:model.Model, stimuli, init_guess, k):
    """
    A wrapped version for two layers
    Output:
    order_params, delw1, delw2

    It solves the self-consistent equations for two-layer networks, and predict the resulting delw matrices.
    """
    def twolayer_selfconsistent(p):
        u1, v1 = p
        Q = (k - a_norm_sq * v1) * u1 * np.eye(W1.shape[1]) + a_norm_sq * W1.T @ W1
        Y = (k - a_norm_sq * v1) * x1- k * W1.T @ W2.T @ a
        Lambda = np.linalg.inv(Q) @ Y
        u1_out = np.linalg.norm((k - a_norm_sq * v1)**-1 * (W2.T @ a * k + a_norm_sq * W1 @ Lambda))**2
        v1_out = np.linalg.norm(Lambda)**2
        return (u1_out-u1) / u1, (v1_out-v1) / v1

    x1 = stimuli.x1_normed.numpy().reshape(-1, 1)

    # Get bounds of the effective matrices
    w_effs, active_inds = get_effective_weights(network, stimuli.x0)
    W1, W2 = w_effs

    forward_mat = W2 @ W1

    a = mse_optimal_a(forward_mat, stimuli)

    a_norm_sq = np.linalg.norm(a)**2

    # Use the fsolve solver to solve self-consistent equations
    u1, v1 = fsolve(twolayer_selfconsistent, init_guess, xtol=1e-15, factor=10, maxfev=100000)

    Q = (k - a_norm_sq * v1) * u1 * np.eye(W1.shape[1]) + a_norm_sq * W1.T @ W1
    Y = (k - a_norm_sq * v1) * x1- k * W1.T @ W2.T @ a
    Lambda = np.linalg.inv(Q) @ Y

    delw1 = get_delw1(Lambda, v1, a_norm_sq, a, W1, W2, k)
    delw2 = get_delw2(Lambda, v1, a_norm_sq, a, W1, W2, k)

    return [u1, v1], delw1, delw2, active_inds, a


def get_del_Ws_threelayer(u1, u2, v1, v2, W1, W2, W3, a, a_norm_sq, x1):
    Q1 = u1 * np.eye(W1.shape[1]) + (1-v1*u2)**-1*u2*W1.T @ W1
    Q2 = (1-a_norm_sq*v2)*np.eye(W2.shape[0])-a_norm_sq*(1-v1*u2)**-1*v1*W2 @ W2.T
    Lambda = np.linalg.inv(Q1 + a_norm_sq*(1-v1*u2)**-2 * W1.T @ W2.T @ np.linalg.inv(Q2) @ W2 @ W1) @ (x1 - (1-v1*u2)**-1 * W1.T @ W2.T @ np.linalg.inv(Q2) @ W3.T @ a)
    U2 = np.linalg.inv(Q2) @ (W3.T @ a + a_norm_sq * (1-v1*u2)**-1 * W2 @ W1 @ Lambda)

    delW1 = (1 - v1*u2)**-1 * (W2.T @ U2 + u2*W1@Lambda)@Lambda.T
    delW2 = (1 - v1*u2)**-1 * U2 @ (v1 * U2.T @ W2 + Lambda.T @ W1.T)
    delW3 = v2 * a @ U2.T + (1-v1*u2)**-1 * a @ (v1 * U2.T @ W2 + Lambda.T @ W1.T) @ W2.T
    return delW1, delW2, delW3


def get_delw1(Lambda, v1, a_norm_sq, a, W1, W2, k):
    return (k - a_norm_sq * v1)**-1 * (k * W2.T @ a + a_norm_sq * W1 @ Lambda) @ Lambda.T

def get_delw2(Lambda, v1, a_norm_sq, a, W1, W2, k):
    return (k - a_norm_sq * v1)**-1 * a @ (v1 * W2.T @ a + W1 @ Lambda).T



def predict_delw_threelayer(network:model.Model, stimuli, init_guess=[100, 100, 0.00005, 0.00005]):
    """
    Solve the order parameter self-consistent equations for three-layer networks.
    """
    def norm_sq(x):
        return np.linalg.norm(x)**2

    x1 = stimuli.x1_normed.numpy().reshape(-1, 1)

    # create the effective weights
    w_effs, active_inds = get_effective_weights(network, stimuli.x0)

    W1, W2, W3 = w_effs

    forward_mat = W3 @ W2 @ W1

    a = mse_optimal_a(forward_mat, stimuli, epsilon=1e-5)
    
    a_norm_sq = np.linalg.norm(a)**2

    # Solve self-consistent equations by iteration
    def iteration_eqs(p):
        u1, u2, v1, v2 = p
        one_minus_v1_u2 = 1 - v1* u2

        one_minus_a_norm_sq_v2 = 1-a_norm_sq*v2

        Q1 = u1 * np.eye(W1.shape[1]) + one_minus_v1_u2**-1*u2*W1.T @ W1
        Q2 = (1-a_norm_sq*v2)*np.eye(W2.shape[0])-a_norm_sq*one_minus_v1_u2**-1*v1*W2 @ W2.T
        Lambda = np.linalg.inv(Q1 + a_norm_sq*one_minus_v1_u2**-2 * W1.T @ W2.T @ np.linalg.inv(Q2) @ W2 @ W1) @ (x1 - one_minus_v1_u2**-1 * W1.T @ W2.T @ np.linalg.inv(Q2) @ W3.T @ a)
        U2 = np.linalg.inv(Q2) @ (W3.T @ a + a_norm_sq * one_minus_v1_u2**-1 * W2 @ W1 @ Lambda)

        u1_out = norm_sq(one_minus_v1_u2**-1 * (W2.T @ U2 + u2 * W1 @ Lambda))
        u2_out = norm_sq(U2)
        v1_out = norm_sq(Lambda)
        v2_out = norm_sq(one_minus_v1_u2**-1 * (W1 @ Lambda + v1 * W2.T @ U2))
        return np.array([u1_out, u2_out, v1_out, v2_out])

    op_vals = [np.array(init_guess)]  # initial guesses for the order parameters
    max_iter = 300

    for i in range(max_iter):
        new_vals = iteration_eqs(op_vals[-1])

        convergence_counter = 0
        for old_op, new_op in zip(op_vals[-1], new_vals):
            if np.abs(new_op - old_op) / np.abs(old_op) < 1e-7:
                convergence_counter += 1

        op_vals.append(0.9 * op_vals[-1] + 0.1 * new_vals)
        if convergence_counter == 4:
            break
        # print(op_vals[-1][0])
    delW1, delW2, delW3 = get_del_Ws_threelayer(*op_vals[-1], W1, W2, W3, a, a_norm_sq, x1)

    return np.array(op_vals), delW1, delW2, delW3, active_inds, a

""" ======================================================================
================================ FIGURE MAKING  ==========================
========================================================================== """


def norm_plot(dataset, loglog=False, ratio=False, layer=0, exit_scatter=False):

    plt.figure(dpi=100)
    ms = []
    x_mins = []; x_maxs = []


    n_layers = dataset['datas'][0]['args'][0].n_layers
    if layer > n_layers - 1:
        print('layer ind exceeds number of layers.')
        return

    for i, collector in enumerate(dataset['collectors']):

        parameter = np.array(collector.get_arg(dataset['ind_var'])).astype(np.float64)
        delW_norm = np.array(dataset['datas'][i]['delW'])[:, layer]

        dims = np.array(collector.get_arg('dims'))

        if loglog:
            x = parameter; y = delW_norm
            if ratio:
                w_norm = np.sqrt(dims[:, layer + 1] / dims[:, layer])
                y /= w_norm
            x = np.log(x); y = np.log(y)

        else:
            plt.axhline(0, color='gray', ls='--')
            x = parameter; y = delW_norm
            if ratio:
                w_norm = np.sqrt(dims[:, layer + 1] / dims[:, layer])
                y /= w_norm
            x_mins.append(x.min()); x_maxs.append(x.max())


        if exit_scatter:
            try:
                codes = np.array(collector.get_arg('exit_code'))
                plt.scatter(x[codes==0], y[codes==0], marker='s', color='C'+str(i), s=20)
            except:
                print('Exit codes not recorded.')
        x, y_mean, y_ste = get_mean_std(x, y)
        plt.errorbar(x, y_mean, y_ste, color='C' + str(i), linestyle='--', label=dataset['label_var'] + '=' +  str(collector.get_arg(dataset['label_var'])[0]), marker='o', markersize=2)

        if loglog:
            m, b = np.polyfit(np.log(parameter), np.log(delW_norm), 1)
            plt.plot(np.log(parameter), np.log(parameter) * m + b, color='k')
            ms.append(m)

    title_text = ''
    for name in dataset['names']:
        title_text += name[13:] + ' \n'

    plt.legend(bbox_to_anchor=(1.05, -0.2))
    if loglog:
        plt.ylabel('log |DelW|'); plt.xlabel('log ' + dataset['ind_var'])
        for m in ms:
            title_text += f'\n m={m:.3f}'
    else:
        plt.ylabel('|DelW|'); plt.xlabel(dataset['ind_var'])
        x_min = np.min(x_mins); x_max = np.max(x_maxs)
        plt.xlim(x_min - (x_max - x_min)*0.1, x_max + (x_max - x_min) * 0.1)


    if ratio:
        if loglog:
            plt.ylabel(' log (|DelW| / |W|)')

        else:
            plt.ylabel('|DelW| / |W|')
    plt.grid()

    title_text +=f'Layer #{layer + 1} \n'
    plt.title(title_text)


def forgetting_plot(dataset, diff_bl=False):

    plt.figure(dpi=100)
    x_mins = []; x_maxs = []
    for i, collector in enumerate(dataset['collectors']):

        _data = dataset['datas'][i]

        parameter = np.array(collector.get_arg(dataset['ind_var'])).astype(np.float64)
        transfer = np.array(_data['transfer'])

        y_min = transfer.min(1); y_max = transfer.max(1)

        x, y_max_mean, y_max_ste = get_mean_std(parameter, y_max)
        plt.errorbar(x, y_max_mean, y_max_ste, color='C' + str(i), label=dataset['label_var'] + '=' +  str(collector.get_arg(dataset['label_var'])[0]), marker='^', markersize=3, alpha=0.7)

        x, y_min_mean, y_min_ste = get_mean_std(parameter, y_min)
        plt.errorbar(x, y_min_mean, y_min_ste, color='C' + str(i), marker='o', markersize=3, alpha=0.7)

        try:
            bl = _data['args'][0]._baseline
        except:
            bl = _data['baseline'][0]
        plt.axhline(bl, color='k', ls='--')
        plt.axhline(bl + np.sqrt(bl*(1-bl)/(2*_data['args'][0].n_test_trials)), color='gray', ls='--')
        x_mins.append(x.min()); x_maxs.append(x.max())

    x_min = np.min(x_mins); x_max = np.max(x_maxs)
    plt.xlim(x_min - (x_max - x_min)*0.1, x_max + (x_max - x_min) * 0.1)
    plt.legend(bbox_to_anchor=(1.05, -0.2))
    plt.axhline(0.035, color='r')


    plt.ylabel('Error frac.'); plt.xlabel(dataset['ind_var'])
    plt.grid()
    title_text = ''
    for name in dataset['names']:
        title_text += name[13:] + ' \n'
    plt.title(str(title_text))


def scaling_fig(D, loglog=False):
    fig, axes = plt.subplots(dpi=100, nrows=2, ncols=1, sharex=True, figsize=(3, 4))
    axes = axes.ravel()
    title_V = ''
    title_r = ''
    for i, collector in enumerate(D['collectors']):
        plt.sca(axes[0])
        if loglog:
            mV, b = fit_and_plot(collector.get_arg(D['ind_var']), collector.get_arg('V_norm'), scatter=True)
            title_V += (D['names'][i][13:] + f' mV={mV:.3f} \n')
        else:
            x, y, ystd = get_mean_std(collector.get_arg(D['ind_var']), collector.get_arg('V_norm'))
            plt.errorbar(x, y, ystd)
            plt.ylabel('|V|')

        plt.title(title_V)

        plt.sca(axes[1])
        if loglog:
            mr, b = fit_and_plot(collector.get_arg(D['ind_var']), collector.get_arg('dr'), scatter=True)
            title_r += (D['names'][i][13:] + f' mr={mr:.3f} \n')
        else:
            x, y, ystd = get_mean_std(collector.get_arg(D['ind_var']), collector.get_arg('dr'))
            plt.errorbar(x, y, ystd)
            plt.ylabel('<dr>')


        plt.title(title_r)

    plt.xlabel(D['ind_var'])
    plt.tight_layout()


def sing_val_fig(D, log=True, layer=0, third=False, ratio=False):
    nrows = 2
    if third:
        nrows += 1
    fig, axes = plt.subplots(dpi=100, ncols=1, nrows=nrows, sharex=True, figsize=(3, 4))
    axes = axes.ravel()

    titles = ['', '']

    for D_ind, collector in enumerate(D['collectors']):
        sing_vals = np.array(D['datas'][D_ind]['sing_val'])[:, layer, :].T

        if ratio:
            sing_vals /= np.sqrt(np.array(collector.get_arg('dims'))[:, layer + 1] / np.array(collector.get_arg('dims'))[:, layer])

        if log:
            for sing_ind in range(2):
                plt.sca(axes[sing_ind])
                m, b = fit_and_plot(collector.get_arg(D['ind_var']), sing_vals[sing_ind], scatter=True)
                titles[sing_ind] += f'  m{sing_ind + 1}={m:.3f} //'; plt.title(titles[sing_ind])
                plt.ylabel(f'log s{sing_ind + 1}')
                plt.grid()

        else:
            for sing_ind in range(2):
                plt.sca(axes[sing_ind])
                x, y_mean, y_ste = get_mean_std(collector.get_arg(D['ind_var']),  sing_vals[sing_ind])
                plt.errorbar(x, y_mean, y_ste, label=D['label_var'] + '=' + str(collector.get_arg(D['label_var'])[0]))
                plt.ylabel(f's{sing_ind + 1}'); plt.axhline(0, color='gray', ls='--'); plt.grid()

                if ratio:
                    plt.ylabel(f'normalized s{sing_ind + 1}')

            if third:
                plt.sca(axes[2])
                x, y_mean, y_ste = get_mean_std(collector.get_arg(D['ind_var']),  sing_vals[layer, 2])
                plt.errorbar(x, y_mean, y_ste,label=D['label_var'] + '=' + str(collector.get_arg(D['label_var'])[0]))
                plt.axhline(0, color='gray', ls='--')
    if log:
        plt.xlabel('log ' + D['ind_var']); plt.legend()
    else:
        plt.xlabel(D['ind_var'])
    plt.legend(bbox_to_anchor=(1, 1))
    plt.suptitle(f'Singular values Layer #{layer + 1}')


def max_firing_fig(response_property):
    slope_fig = plt.figure(figsize=(5, 4))
    max0 = np.repeat(np.array(response_property['max0']).reshape(-1, 1), len(response_property['max'][0]), 1)
    normalized_del_max = (np.array(response_property['max']) - max0) / max0 * 100
    vmax = np.max(np.abs(normalized_del_max)); vmin = -vmax
    plt.imshow(normalized_del_max, extent=(0, 4, 0, 2), cmap='coolwarm', vmax=vmax, vmin=vmin)
    cbar = plt.colorbar(orientation='horizontal')
    cbar.set_label('% change from pretraining level')
    plt.yticks([0, 1, 2], [1, 0.55, 0.1]); plt.ylabel('$\sigma_w$')
    plt.xlabel('neuron index', labelpad=-5)
    plt.xticks([0, 4], [1, len(response_property['max'][0])])


def mean_firing_fig(response_property):
    slope_fig = plt.figure(figsize=(5, 4))
    mean0 = np.repeat(np.array(response_property['mean0']).reshape(-1, 1), len(response_property['mean'][0]), 1)
    normalized_del_mean = (np.array(response_property['mean']) - mean0) / mean0 * 100
    vmax = np.max(np.abs(normalized_del_mean)); vmin = -vmax
    plt.imshow(normalized_del_mean, extent=(0, 4, 0, 2), cmap='coolwarm', vmax=vmax, vmin=vmin)
    cbar = plt.colorbar(orientation='horizontal')
    cbar.set_label('% change from pretraining level')
    plt.yticks([0, 1, 2], [1, 0.55, 0.1]); plt.ylabel('$\sigma_w$')
    plt.xlabel('Stimulus', labelpad=-5)
    plt.xticks([0, 4], [1, 6.28])


def slope_at_trained_stimulus_figure(response_property):
    slope_fig = plt.figure()

    del_slope = np.abs(response_property['slope']) - np.abs(response_property['slope0'])
    del_slope /= np.abs(response_property['slope0']).mean(1, keepdims=True)
    vmax = np.max(np.abs(del_slope)); vmin = -vmax
    plt.imshow(del_slope, extent=(0, 4, 0, 2), cmap='coolwarm', vmax=vmax, vmin=vmin)
    cbar = plt.colorbar(orientation='horizontal')
    cbar.set_label('$\Delta$ slope / population mean slope before training')
    plt.yticks([0, 1, 2], [1, 0.55, 0.1]); plt.ylabel('$\sigma_w$')
    plt.xlabel('neuron index', labelpad=-5)
    plt.xticks([0, 4], [1, del_slope.shape[1]])


def relative_slope_at_trained_stimulus_figure(response_property):
    """
    It's slope at the trained angle over the sqrt of the average activity
    """
    relaitve_slope_fig = plt.figure(figsize=(4, 3))
    sqrt_mean_over_stim = np.sqrt(np.array(response_property['mean_over_stim']))
    sqrt_mean_over_stim0 = np.sqrt(np.repeat(np.array(response_property['mean0']).reshape(-1, 1), sqrt_mean_over_stim.shape[1], axis=1))

    del_slope = np.abs(response_property['slope']) / sqrt_mean_over_stim - np.abs(response_property['slope0']) / sqrt_mean_over_stim0
    del_slope /= (np.abs(response_property['slope0']) / sqrt_mean_over_stim0).mean(1, keepdims=True)

    vmax = np.max(np.abs(del_slope)); vmin = -vmax
    plt.imshow(del_slope, extent=(0, 4, 0, 2), cmap='coolwarm', vmax=vmax, vmin=vmin)
    cbar = plt.colorbar(orientation='horizontal')
    cbar.set_label('relative slope: $\Delta$ / population mean')
    plt.yticks([0, 1, 2], [1, 0.55, 0.1]); plt.ylabel('$\sigma_w$')
    plt.xlabel('neuron index', labelpad=-5)
    plt.xticks([0, 4], [1, sqrt_mean_over_stim.shape[1]]); plt.title('First layer')
    

def preferred_orientation_histogram_figure(response_properties, nbins=50):
    plt.figure(figsize=(5, 4))
    pref_histograms = []
    for pref in response_properties['pref']:
        hist, bin_edges = np.histogram(pref, bins=np.linspace(-0.1, pref.shape[0]-.5, nbins+1))
        pref_histograms.append(hist)
    data = (np.array(pref_histograms) - pref.shape[0] / nbins) / (pref.shape[0] / nbins)
    vmax = np.max(np.abs(data)); vmin = -vmax
    plt.imshow(data, extent=(0, 2*np.pi, 0, 5), cmap='coolwarm', vmax=vmax, vmin=vmin); plt.xlabel('Theta', labelpad=-1)
    cbar = plt.colorbar(orientation='horizontal')
    plt.yticks([0, 2.5, 5], [1, 0.55, 0.1]); plt.ylabel('$\sigma_w$')
    cbar.set_label('$\Delta$ #neurons per bin/#neurons per bin')


def single_neuron_snr_figure(response_properties, l):
    init_vecs = np.array(response_properties[l]['snr0'])
    new_vecs = np.array(response_properties[l]['snr'])
    normalized_del_snr = (new_vecs - init_vecs) / init_vecs.mean(axis=1, keepdims=True)

    
    plt.figure(figsize=(4, 3)) 
    vmax = np.max(np.abs(normalized_del_snr)); vmin=-vmax
    plt.imshow(normalized_del_snr, extent=(0, 4, 0, 2), cmap='coolwarm', vmax=vmax, vmin=vmin)
    plt.yticks([0, 1, 2], [1, 0.55, 0.1]); plt.ylabel('$\sigma_w$')
    cbar = plt.colorbar(orientation='horizontal')
    cbar.set_label('$\Delta$ SNR / population mean init SNR')
    plt.xlabel('neuron index', labelpad=-5)
    plt.xticks([0, 4], [1, new_vecs.shape[1]])
    plt.tight_layout()


def single_neuron_noise_figure(response_properties, l):
    init_vecs = np.array(response_properties[l]['noise0'])
    new_vecs = np.array(response_properties[l]['noise'])
    normalized_del_noise = (new_vecs - init_vecs) / init_vecs.mean(axis=1, keepdims=True)
    lim = np.abs(normalized_del_noise).max()
    plt.figure(figsize=(4, 3)) 
    plt.imshow(normalized_del_noise, extent=(0, 4, 0, 2), cmap='coolwarm', vmin=-lim, vmax=lim)
    plt.yticks([0, 1, 2], [1, 0.55, 0.1]); plt.ylabel('$\sigma_w$')
    cbar = plt.colorbar(orientation='horizontal')
    cbar.set_label('$\Delta$ noise / mean init noise')
    plt.xlabel('neuron index', labelpad=-5)
    plt.xticks([0, 4], [1, init_vecs.shape[1]])
    plt.tight_layout()
    

def fi_transfer_fig(fi_arr, fi0_arr):

    plt.figure(figsize=(4, 3))
    relative_del_fi = (fi_arr / fi0_arr - 1) * 100
    map_norm = mpl.colors.DivergingNorm(0)

    plt.imshow(relative_del_fi, cmap='coolwarm', norm=map_norm)
    cbar = plt.colorbar()
    cbar.set_label('% $ \Delta FI$ from baseline')
    plt.xlabel('Stimulus'); plt.xticks([0, 16, 31], ['0', 'pi', '2pi'])
    plt.ylabel('$\sigma_w$'); plt.yticks([0, fi_arr.shape[0]-1], [0.1, 1])


def error_transfer_fig(fi_arr, fi0_arr):
    error_array = fi_to_error(fi_arr)
    baseline_error_array = fi_to_error(fi0_arr)

    plt.figure(figsize=(4, 3))
    norm = mpl.colors.DivergingNorm(vcenter=0)
    plt.imshow(error_array - baseline_error_array, cmap='coolwarm', norm=norm)
    cbar = plt.colorbar()
    cbar.set_label('$\Delta$ error')
    plt.xlabel('Stimulus'); plt.xticks([0, 16, 31], ['0', 'pi', '2pi'])
    plt.ylabel('$\sigma_w$'); plt.yticks([0, error_array.shape[0]-1], [0.1, 1])


def make_w_fig(layer_ind, data, delw1_list, delw2_list, delw3_list, active_inds_list, sig_w_array, theory_ind):

    w_fig = plt.figure()
    w_fig.add_subplot(121); plt.title(f'Theory $\sigma_w=${sig_w_array[theory_ind]:.3f}')
    if layer_ind == 0:
        image = delw1_list[theory_ind]
    elif layer_ind == 1:
        image = delw2_list[theory_ind]
    elif layer_ind == 2:
        image = delw3_list[theory_ind]

    vmax = np.max(np.abs(image * data['args'].N))
    img = plt.imshow(image * data['args'].N, cmap='coolwarm', vmax=vmax, vmin=-vmax)

    w_fig.add_subplot(122)
    active_inds = active_inds_list[theory_ind]
    plt.imshow(recover_matrix(data['delW'][0][-1])[active_inds[layer_ind+1]][:, active_inds[layer_ind]] * data['args'].N, cmap='coolwarm', vmax=vmax, vmin=-vmax)
    plt.title(f'Numerical, $\sigma_w=$ {data["args"].sig_w:.3f}')
    cbar_ax = w_fig.add_axes([0.25, 0.1, 0.5, 0.05])
    w_fig.colorbar(img, cax=cbar_ax, orientation='horizontal')
    plt.tight_layout()

""" ======================================================================
================================ ANALYZE NETWORKS  ==========================
========================================================================== """

def get_phiprime(network:model.Model, args, to_layer):
    mean_input = torch.from_numpy(make_x0(args.theta, args.sig_s, args.N).T).float()
    input = (torch.normal(mean=torch.zeros(args.N, 1000))*np.sqrt(args.noise_var) + mean_input.t()).t()
    obs = network.get_obs(input, to_layer=to_layer + 1)  # convert zero indexing to one indexing
    return (obs > 0).float().mean(0).numpy().T



def get_v_eff(network:model.Model, phi_prime:list, a:list, time_index:int):
    W_tildes = []
    for i, W in enumerate(network.Ws):
        W_mean_sub = W - W.mean()
        W_tildes.append(phi_prime[i][time_index].reshape(-1, 1).repeat(network.N, 1) * W_mean_sub)

    W_tildes.reverse()  # flip the order of weights
    v_eff = a[time_index]
    for w_tilde in W_tildes:
        v_eff = w_tilde.T @ v_eff

    return v_eff



""" ======================================================================
================================ GENERAL  ==========================
========================================================================== """


""" Other utils """

def get_cos(vec_list):
    dir = []
    for vec in vec_list:
        dir.append(cos(vec, vec_list[0]))
    return dir

def get_changes(list_of_vecs):
    new_list = [np.array(v).flatten() for v in list_of_vecs]
    result = [(np.linalg.norm(np.array(v).flatten() - new_list[0]) / np.linalg.norm(new_list[0])) for v in new_list]
    return np.array(result)

def decompose_matrix(matrix, ranks):
    """Performs SVD on matrices and save the first n-ranks."""
    u, s, v = np.linalg.svd(np.array(matrix))
    sing_vals = s[:ranks]
    lefts = u[:, :ranks]
    rights = v[:ranks, :]
    return {'u':lefts, 's':sing_vals, 'v':rights}


def recover_matrix(decomposed_matrix):
    try:
        return decomposed_matrix['u'] @ np.diag(decomposed_matrix['s']) @ decomposed_matrix['v']
    except:
        return decomposed_matrix[1] @ np.diag(decomposed_matrix[0]) @ decomposed_matrix[2]


def theory_thetas(noise_var, delta, loss, theta1_0, theta2_0, length, coefs):

    """interpolate coefs to get more coefs"""
    interpolated = inter.interp1d(range(len(coefs)), coefs)

    denser_x = np.linspace(0, len(coefs) - 1, 100)
    warping_factor = length / 100

    interpolated_coefs = interpolated(denser_x)
    """Calculate theoretical trajectories of theta1, theta2"""
    assert loss in ['bce', 'mse']

    axis = np.linspace(0, length - 1, length)
    theta1s = np.zeros(len(axis)); theta2s = np.zeros(len(axis))
    theta1s[0] = theta1_0
    theta2s[0] = theta2_0

    if loss == 'bce':
        for i in range(len(axis) - 1):
            fields = np.random.normal(-theta1s[i]*delta*np.ones(10000), scale=np.sqrt(noise_var*(theta1s[i]**2 + theta2s[i]**2)))
            sig_field = 1 / (1 + np.exp(-fields)); f1 = np.mean(sig_field); f2 = np.mean(sig_field - sig_field**2)
            dtheta1 = delta * f1 - noise_var * f2 * theta1s[i]
            dtheta2 = -noise_var * f2 * theta2s[i]
            theta1s[i+1] = theta1s[i] + dtheta1 * coefs[i]; theta2s[i+1] = theta2s[i] + dtheta2 * coefs[i]
        return axis, theta1s, theta2s

    elif loss == 'mse':
        for i in range(len(axis) - 1):
            dtheta1 = -(delta**2 + noise_var) * (theta1s[i] - 1)*2
            dtheta2 = -noise_var * theta2s[i]*2
            theta1s[i+1] = theta1s[i] + dtheta1 * coefs[i]; theta2s[i+1] = theta2s[i] + dtheta2 * coefs[i]
        # theta1s = 1 + (theta1_0 - 1) * np.exp(-np.mean(coef) * axis * (delta**2 + noise_var) * 2)
        # theta2s = np.exp(-np.mean(coef) * axis * noise_var * 2) * theta2_0
        return axis, theta1s, theta2s

def get_mean_std(x, y):
    _x = np.array(x).copy(); _y = np.array(y).copy()
    unique = np.unique(_x.flatten())
    mean = np.zeros(len(unique))
    ste = np.zeros_like(mean)

    for i, unique_x in enumerate(unique):
        ys = _y[_x == unique_x]
        mean[i] = ys.mean()
        ste[i] = ys.std() / np.sqrt(len(ys))
    return unique, mean, ste


def compare_sing_vals(D, D_ind, layer):
    plt.figure(dpi=100)
    for i in range(3):
        x, ymean, ystd = get_mean_std(D['collectors'][D_ind].get_arg(D['ind_var']), np.log10(np.array(D['datas'][D_ind]['sing_val'])[:, layer, i]))
        plt.errorbar(x, ymean, ystd, label=f'#{i+1}')
    plt.xlabel(D['ind_var'])
    plt.ylabel('log10 sing vals.')
    plt.title(f'Singular Values from Layer {layer + 1}, ' + D['names'][D_ind][13:-10])
    plt.legend(); plt.grid()


def show_exit_code(dataset, C_list, ind_var, D_ind=0):


    collector = dataset['collectors'][D_ind]
    name = dataset['names'][D_ind]

    if 'exit_code' in vars(collector.data[0]['args']).keys():

        exit, axes = plt.subplots(nrows=2, ncols=1, sharex=True, dpi=100, figsize=(5, 3))
        axes = axes.ravel()
        plt.sca(axes[0])
        plt.scatter(collector.get_arg(ind_var), collector.get_arg('exit_code'), alpha=0.5)
        plt.yticks([0, 1, 2], ['MLD', 'Conv.', 'Time out'])
        plt.grid()
        plt.title('Exit codes')
        plt.sca(axes[1])
        plt.scatter(collector.get_arg(ind_var), collector.get_arg('_duration') / 3600)
        plt.xlabel(ind_var)
        plt.title('Run Duration (hrs)')
        plt.grid()
        plt.tight_layout()
        plt.suptitle(name[13:])
        plt.subplots_adjust(top=0.83)
    else:
        print('Exit codes not recorded.')


def fit_and_plot(x, y, scatter=False, label=None):
    m, b = np.polyfit(np.log(x), np.log(y), 1)
    plt.plot(np.log(x), np.log(x)*m+b, color='k')
    if scatter:
        _x, _y, _y_std = get_mean_std(np.log(x), np.log(y))
        # plt.errorbar(_x, _y, _y_std, label=label)
        plt.scatter(np.log(x), np.log(y), label=label)
    return m, b

def load_theory_from_file(net, theory_dict, ind):
    delw1 = theory_dict['delw1']
    if net.n_layers > 1:
        delw2 = theory_dict['delw2']
    if net.n_layers > 2:
        delw3 = theory_dict['delw3']
    act_inds = theory_dict['active_inds']

    if net.n_layers == 1:
        load_theory_changes(net, [delw1[ind]], act_inds[ind])
    elif net.n_layers == 2:
        load_theory_changes(net, [delw1[ind], delw2[ind]], act_inds[ind])
    elif net.n_layers == 3:
        load_theory_changes(net, [delw1[ind], delw2[ind], delw3[ind]], act_inds[ind])

def load_theory_changes(network:model.Model, delw_list:list, active_inds):

    full_delws = get_full_delw_mats(delw_list, active_inds, network)

    for _W, delw in zip(network.Ws, full_delws):
        _W += delw

def mse_optimal_a(forward_mat, stimuli, sing_val_truncation=1):
    """
    format_mat: W^L @ W^L-1 ... @ W^1. Must be a numpy array.

    Return a numpy array. Column vector.
    """
    x1 = stimuli.x1_normed.t().cpu().numpy()
    F_u, F_s, F_v = np.linalg.svd(forward_mat)

    '''Calculate a coefficient'''
    xvsq = np.linalg.norm(F_v[:sing_val_truncation + 1] @ x1)**2
    delta = np.linalg.norm(stimuli.x1.cpu().numpy()) # this delta is not delta theta
    coef = (1 + delta**2 / stimuli.noise_var) / (1 + delta**2 / stimuli.noise_var * xvsq)

    '''Compute the direction'''
    inv_noise = np.linalg.pinv(forward_mat @ forward_mat.T, rcond=(F_s**2)[sing_val_truncation] / (F_s**2)[0] - 0.001)
    a = inv_noise @ forward_mat @ x1
    return a * coef

def test_linear(model, stimuli, max_iter=50000, return_weights=False, C=1, to_layer=None):


    with torch.no_grad():
        obs = model.get_obs(stimuli.te_stim(), to_layer=to_layer).numpy()

    linear_tester = LinearSVC(max_iter=max_iter, C=C)
    linear_tester.fit(obs, stimuli.test_labels.numpy().flatten())
    error = 1-linear_tester.score(obs, stimuli.test_labels.numpy().flatten())

    if return_weights:
        return error, linear_tester.coef_, linear_tester.intercept_
    else:
        return error

def test_at_angles(network:model.Model, args, npoints=16, show_bar=True, C=1, to_layer=None, device=None):
    """
    Using Linear SVM as the probe, test performance of the network at different orientations with a *relearned* readout.
    Only works on CPUs.
    """
    net_cpu = network.cpu()
    angles = np.linspace(0, np.pi*2, npoints, endpoint=False)
    if show_bar:
        print('\n')
    tic = time.time()
    error_array = np.zeros_like(angles)
    _curr_theta = args.theta  # store current value
    _curr_n_test = args.n_test_trials  # store current number of test trials
    for ind, angle in enumerate(angles):
        args.theta = angle; args.n_test_trials = 50000
        temp_stim = GaborStimuli(args, verbose=False, device=None)
        error_array[ind] = test_linear(net_cpu, temp_stim, C=C, to_layer=to_layer)
        if show_bar:
            print((ind + 1) * '||' + (len(angles)-1-ind) * '00' + f'  {int(time.time()-tic)} secs', end='\r')

    # restore stored value
    args.theta = _curr_theta; args.n_test_trials = _curr_n_test
    return angles, error_array

def test_at_angles_gpu(network:model.Model, manager, args, npoints=16, show_bar=True, to_layer=None, device=None):
    """
    Using a linear classifier as the probe. Runs on the gpu. Leverages the pretrain_readout function.
    """
    angles = np.linspace(0, np.pi*2, npoints, endpoint=False)
    if show_bar:
        print('\n')

    error_array = np.zeros_like(angles)
    _curr_theta = args.theta  # store current value
    _curr_ro = network.RO.weight.data.clone()  # saves the current readout vector

    torch.set_grad_enabled(True)  # from now on, autograd is no longer needed

    # Turn gradients off for network weights
    for L in network.Ls:
        L.weight.requires_grad = False

    for ind, angle in enumerate(angles):

        # create a different stimulus
        args.theta = angle
        temp_stim = GaborStimuli(args, verbose=False, device=device)

        # the pretrain_readout function already initializes the readout
        error_array[ind] = pretrain_readout(network, manager,
                                            temp_stim, args.loss, return_error=True,
                                             error_converge=True, force_gd=True)

    # restore stored value
    args.theta = _curr_theta
    network.RO.weight.data = _curr_ro
    return angles, error_array


def test_at_angles_meanfield(network:model.Model, args, npoints=16, show_bar=True, to_layer=None, device=None):
    """
    Using a linear classifier as the probe. Runs on the gpu. Leverages the pretrain_readout function.
    """
    angles = np.linspace(0, np.pi*2, npoints, endpoint=False)
    if show_bar:
        print('\n')

    error_array = np.zeros_like(angles)
    _curr_theta = args.theta  # store current value
    _curr_ro = network.RO.weight.data.clone()  # saves the current readout vector

    torch.set_grad_enabled(True)  # from now on, autograd is no longer needed

    # Turn gradients off for network weights
    for L in network.Ls:
        L.weight.requires_grad = False

    for ind, angle in enumerate(angles):

        # create a different stimulus
        args.theta = angle
        temp_stim = GaborStimuli(args, verbose=False, device=device)

        fi = get_mean_field_fi(network, temp_stim)

        # v = temp_stim.x1_normed.t()

        # eff_weights = get_effective_weights(network, temp_stim.x0, full_mat=True)[0]
        # forward_mat = eff_weights[0]
        # for i in range(len(eff_weights)):
        #     if i > 0:
        #         forward_mat = eff_weights[i] @ forward_mat
        
        # covar = torch.FloatTensor(forward_mat @ forward_mat.T)
        # _u, _s, _v = torch.svd(covar)
        # a = torch.inverse(covar + 1e-7 * torch.eye(covar.shape[1])).to(network.RO.weight.device) @ torch.FloatTensor(forward_mat).to(network.RO.weight.device) @ v
        # network.RO.weight.data = a.view(1, -1)
        
        # te_stim = temp_stim.te_stim()
        # net_output = network(te_stim)
        # print(net_output.shape)
        # te_error = float(torch.mean((net_output * (2*temp_stim.test_labels-1) < 0).float()).cpu())


        # the pretrain_readout function already initializes the readout
        error_array[ind] = fi_to_error(fi)

    # restore stored value
    args.theta = _curr_theta
    network.RO.weight.data = _curr_ro
    return angles, error_array

def test_transfer_with_original_readouts(network:model.Model, changes_to_weights:list, stimuli:GaborStimuli, npoints=100, n_replicas=40):
    """
    changes_to_weights: list of torch tensors corresponding to changes to weight matrices
    network: model object
    npoints: number of points to sample (i.e. 100 is to spread 100 points across 0 2pi)
    n_replicas: number of mem_saving_avgs to do.
    """
    errors = np.zeros(npoints)

    for i in range(npoints):
        network.RESET_WEIGHTS()
        for _L, delw in zip(network.Ls, changes_to_weights):
            step_size = int(_L.weight.shape[0] / npoints)
            _L.weight.data += torch.roll(torch.roll(delw, i*step_size, dims=0), i*step_size, dims=1)

        for mem_saving_avg in range(n_replicas):
            _te_error = test(network, stimuli, loss_type='BCE')[0]
            errors[i] += _te_error / n_replicas
    return errors


def fit_tuning_curves(response_mat):
    """
    Fit tuning curves with a circulant Gaussian curve. response_mat should be #angles * # neurons. The angles should be evenly spaced between 0 to 2pi. Returns four fitted parameters for each neuron (#neurons * 4)
    *Requires curve_fit from scipy.optimize, and the tuning curve function.
    """
    tuning_params = np.zeros((response_mat.shape[1], 4))
    tested_angles = np.linspace(0, 2*np.pi, response_mat.shape[0], endpoint=False)
    for i in range(response_mat.shape[1]):
        tuning_params[i] = curve_fit(tuning_curve, tested_angles, response_mat[:, i], p0=(0, 0.05, 0.5, tested_angles[np.argmax(response_mat[:, i])]))[0]
    return tuning_params

def get_effective_weights_with_inds(network:model.Model, active_inds):
    """
    Get the effective matrix of a W, given indices of active neurons in the two layers.
    Convention: active_inds of the input layer is included as np.arange(N)
    Return the full effective matrices (i.e., of same dimensions as the original Ws)
    """
    assert network.n_layers + 1 == len(active_inds)
    output_list = []
    for l in range(network.n_layers):
        full_w = network.Ls[l].weight.data.numpy()
        mask = np.zeros_like(full_w)
        mask[active_inds[l+1]] = 1
        output_list.append(mask * full_w)
    return output_list


def get_response_mats(network:model.Model, to_layer, stim_width, stim_dim, threshold=0):
    """
    The response matrix has dimension n_angles * n_neurons
    """
    probe_stimuli = [make_x0(test_theta, stim_width, stim_dim).flatten() for test_theta in np.linspace(0, 2*np.pi, stim_dim, endpoint=False)]
    return network.get_obs(torch.from_numpy(np.array(probe_stimuli)).float(), to_layer=to_layer, threshold=threshold).data.numpy()


def get_linear_obs(effective_weights:list, input_tensor, to_layer):
    output = np.array(input_tensor).copy()
    for i in range(to_layer):
        output = output @ effective_weights[i].T
    return output

def get_mean_field_fi(network:model.Model, stimuli, epsilon=1e-6, to_layer=None, threshold=0):
    """
    Compute the linear Fisher Information of a stimulus in the last layer of the network, at the mean field limit.
    epsilon: small constant to stabilize covariance matrix inversion
    """
    
    if to_layer is None:
        to_layer = network.n_layers
    
    effective_Ws = get_effective_weights(network, stimuli.x0, full_mat=False, threshold=threshold)[0]
    forward_mat = effective_Ws[0]
    for i in np.arange(1, to_layer):
        forward_mat = effective_Ws[i] @ forward_mat
    slope = forward_mat @ stimuli.x1.cpu().t().numpy()
    covar = forward_mat @ forward_mat.T * stimuli.noise_var
    if covar.shape[0] == 0:
        return 0
    else:
        u, s, v = np.linalg.svd(covar)
        return float(slope.T @ np.linalg.inv(covar + s[0] * epsilon * np.eye(covar.shape[0])) @ slope)

def get_mean_field_fi_sing_val(network:model.Model, stimuli, truncate=2, to_layer=None):
    """
    Compute the linear Fisher Information of a stimulus in the last layer of the network, at the mean field limit.
    epsilon: small constant to stabilize covariance matrix inversion
    """
    
    if to_layer is None:
        to_layer = network.n_layers
    
    effective_Ws = get_effective_weights(network, stimuli.x0, full_mat=False)[0]
    forward_mat = effective_Ws[0]
    for i in np.arange(1, to_layer):
        forward_mat = effective_Ws[i] @ forward_mat
    slope = forward_mat @ stimuli.x1.cpu().t().numpy()
    covar = forward_mat @ forward_mat.T * stimuli.noise_var
    u, s, v = np.linalg.svd(covar)
    return float(slope.T @ np.linalg.pinv(covar, rcond=s[truncate] / s[0] - 1e-5) @ slope)

def get_full_delw_mats(delw_list, active_inds, network):
    """
    Using the active_inds, make full delW matrices from the delW on effective matrices.
    """
    full_delw_list = []
    for i in range(len(delw_list)):

        partial_delw = np.zeros((len(active_inds[i+1]), network.Ls[i].weight.shape[1]))
        partial_delw[:, active_inds[i]] = delw_list[i]
        full_delw = np.zeros(network.Ls[i].weight.shape)
        full_delw[active_inds[i+1]] = partial_delw
        full_delw_list.append(full_delw)
    return full_delw_list

def compute_response_properties(init_network:model.Model, trained_network:model.Model, stimuli, response_properties):
    """
    Wrapper function for generating a list of dictionaries of response properties.
    Collect the following properties:
    pref: preferred orientation of each unit (#sig_w * #neurons)
    slope: slope of the tuning curve of each unit at the trained stimulus (#sig_w * #neurons)
    max: max activity of each neuron across stimuli (#sig_w * #neurons). These are already divided by the max activity of pretrained neurons
    mean: mean (across neurons) activity for each angle (#sig_w * #angles). These are already divided by the max activity of pretrained neurons
    baseline mean: same as mean, before training (stored as a scalar)
    baseline max: same as max, before training (stored as a scalar)

    Note: norm of untrained matrices is the same regardless of sig_w
    
    """
    def get_noise_vec(list_of_weights, return_covar=False):

        if len(list_of_weights) == 3:
            covar1 = list_of_weights[0] @ list_of_weights[0].T
            covar2 = list_of_weights[1] @ list_of_weights[0] @ list_of_weights[0].T @ list_of_weights[1].T
            covar3 = list_of_weights[2] @ list_of_weights[1] @ list_of_weights[0] @ list_of_weights[0].T @ list_of_weights[1].T @ list_of_weights[2].T
            covars = [covar1, covar2, covar3]
        
        elif len(list_of_weights) == 2:
            covar1 = list_of_weights[0] @ list_of_weights[0].T
            covar2 = list_of_weights[1] @ list_of_weights[0] @ list_of_weights[0].T @ list_of_weights[1].T
            covars = [covar1, covar2]
        
        elif len(list_of_weights) == 1:
            covar1 = list_of_weights[0] @ list_of_weights[0].T
            covars = [covar1]

        if return_covar:
            return [np.diag(c) for c in covars], covars
        else:
            return [np.diag(c) for c in covars]

    def get_slope_at_trained_stim(list_of_weights, x1):
        x1_np = np.array(x1).T
        if len(list_of_weights) == 3:
            return [list_of_weights[0] @ x1_np, list_of_weights[1] @ list_of_weights[0] @x1_np, list_of_weights[2] @ list_of_weights[1] @ list_of_weights[0] @ x1_np / stimuli.delta]
        elif len(list_of_weights) == 2:
            return [list_of_weights[0] @ x1_np, list_of_weights[1] @ list_of_weights[0] @x1_np / stimuli.delta]
        elif len(list_of_weights) == 1:
            return [list_of_weights[0] @ x1_np / stimuli.delta]


    # get the active inds for the standard stimulus
    active_inds = get_active_inds(init_network, stimuli.x0)

    # Make a list of response matrices using initial (pretraining) weights
    responses0 = [get_response_mats(init_network, to_layer=i+1, stim_width=stimuli.sig_s, stim_dim=init_network.N) for i in range(init_network.n_layers)]

    # get initial effective weights (for the standard stimulus)
    init_eff_weights = get_effective_weights_with_inds(init_network, active_inds)

    # get initial effective weights (for the standard stimulus)
    eff_weights = get_effective_weights_with_inds(trained_network, active_inds)

    # Make a list of response matrices using trained weights
    responses = [get_response_mats(trained_network, to_layer=l+1, stim_width=stimuli.sig_s, stim_dim=trained_network.N) for l in range(trained_network.n_layers)]

    noise0_list, covar0_list = get_noise_vec(init_eff_weights, return_covar=True)
    noise_list, covar_list = get_noise_vec(eff_weights, return_covar=True)

    slope0_list = get_slope_at_trained_stim(init_eff_weights, stimuli.x1)
    slope_list = get_slope_at_trained_stim(eff_weights, stimuli.x1)

    for l in range(trained_network.n_layers):

        preferred_orientation = np.argmax(responses[l], axis=0)
        max_firing = np.max(responses[l], axis=0)
        population_mean = np.mean(responses[l], axis=1)

        response_properties[l]['mean0'].append(responses0[l].mean())
        response_properties[l]['max0'].append(responses0[l].max())
        response_properties[l]['mean_over_stim'].append(responses[l].mean(axis=0))
        response_properties[l]['snr0'].append(slope0_list[l].flatten()**2 / (noise0_list[l] + 1e-7))
        response_properties[l]['snr'].append(slope_list[l].flatten()**2 / (noise_list[l] + 1e-7))
        response_properties[l]['noise0'].append(noise0_list[l])
        response_properties[l]['noise'].append(noise_list[l])
        response_properties[l]['slope'].append(slope_list[l].flatten())
        response_properties[l]['slope0'].append(slope0_list[l].flatten())
        response_properties[l]['pref'].append(preferred_orientation)
        response_properties[l]['max'].append(max_firing)
        response_properties[l]['mean'].append(population_mean)
        response_properties[l]['covar'].append(covar_list[l])
        response_properties[l]['covar0'].append(covar0_list[l])
        # response_properties[l]['tuning_params'].append(fit_tuning_curves(responses[l]))
        # response_properties[l]['tuning_params0'].append(fit_tuning_curves(responses0[l]))

