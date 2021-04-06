
class SphereStimuli:
    def __init__(self, args):

        self.N = args.N
        self.radius_frac = args.radius_frac
        self.n_test_trials = args.n_test_trials
        self.n_train_trials = args.n_train_trials

        self.x0p = make_x0(args.theta + args.delta, args.sig_s, args.N).reshape((1, args.N))
        self.x0m = make_x0(args.theta - args.delta, args.sig_s, args.N).reshape((1, args.N))

        self.center_distance = np.linalg.norm(self.x0p - self.x0m)
        print(f'Center distance: {self.center_distance}')

        self.radius = self.center_distance * self.radius_frac

        self.train_labels = \
        torch.cat((torch.ones(args.n_train_trials, 1), torch.zeros(args.n_train_trials, 1)), dim=0)
        self.test_labels = \
        torch.cat((torch.ones(args.n_test_trials, 1), torch.zeros(args.n_test_trials, 1)), dim=0)
        self.mld_err = 0
        self.noise_var = 0

    def _add_sphere_noise(self, n_points, mean1, mean2):

        spheres1 = np.random.normal(size=(n_points, self.N))
        norm_sphere1 = (spheres1.T / np.linalg.norm(spheres1, axis=1)).T
        spheres2 = np.random.normal(size=(n_points, self.N))
        norm_sphere2 = (spheres2.T / np.linalg.norm(spheres2, axis=1)).T
        return np.concatenate((mean1 + self.radius * norm_sphere1, mean2 + self.radius * norm_sphere2), axis=0)

    def train_mean(self):
        return torch.from_numpy(self._add_sphere_noise(self.n_train_trials, self.x0p, self.x0m)).float()

    def test_mean(self):
        return torch.from_numpy(self._add_sphere_noise(self.n_test_trials, self.x0p, self.x0m)).float()


class LowDimSphere:
    def __init__(self, args, v0=None):

        self.N = args.N
        self.dim = args.sphere_dim; assert self.dim <= self.N
        self.radius_frac = args.radius_frac
        self.n_test_trials = args.n_test_trials
        self.n_train_trials = args.n_train_trials

        self.x0p = make_x0(args.theta + args.delta, args.sig_s, args.N).reshape((1, args.N))
        self.x0m = make_x0(args.theta - args.delta, args.sig_s, args.N).reshape((1, args.N))

        _rand_mat = np.random.normal(size=(args.N, args.N))
        _rand_mat[:, 0] = self.x0p - self.x0m
        q, r = np.linalg.qr(_rand_mat)
        self.proj_mat = q[:, :self.dim].T

        self.center_distance = np.linalg.norm(self.x0p - self.x0m)
        # print(f'Center distance: {self.center_distance}')

        self.radius = self.center_distance * self.radius_frac * np.sqrt(self.dim / self.N)

        self.train_labels = \
        torch.cat((torch.ones(args.n_train_trials, 1), torch.zeros(args.n_train_trials, 1)), dim=0)
        self.test_labels = \
        torch.cat((torch.ones(args.n_test_trials, 1), torch.zeros(args.n_test_trials, 1)), dim=0)
        self.mld_err = 0
        self.noise_var = 0

    def _add_sphere_noise(self, n_points, mean1, mean2):

        sphere1_low = np.random.normal(size=(n_points, self.dim))
        spheres1 = sphere1_low @ self.proj_mat
        norm_sphere1 = (spheres1.T / np.linalg.norm(spheres1, axis=1)).T
        sphere2_low = np.random.normal(size=(n_points, self.dim))
        spheres2 = sphere2_low @ self.proj_mat
        norm_sphere2 = (spheres2.T / np.linalg.norm(spheres2, axis=1)).T
        return np.concatenate((mean1 + self.radius * norm_sphere1, mean2 + self.radius * norm_sphere2), axis=0)

    def train_mean(self):
        return torch.from_numpy(self._add_sphere_noise(self.n_train_trials, self.x0p, self.x0m)).float()

    def test_mean(self):
        return torch.from_numpy(self._add_sphere_noise(self.n_test_trials, self.x0p, self.x0m)).float()


class ManifoldStimuli:
    def __init__(self, args):

        self.N = args.N
        self.sig_s1 = args.sig_s
        self.sig_s2 = args.sig_s + args.del_sig_s
        self.n_test_trials = args.n_test_trials
        self.n_train_trials = args.n_train_trials

        self.input = self._sample_points(self.n_test_trials)

        self.train_labels = \
        torch.cat((torch.ones(args.n_train_trials, 1), torch.zeros(args.n_train_trials, 1)), dim=0)
        self.test_labels = \
        torch.cat((torch.ones(args.n_test_trials, 1), torch.zeros(args.n_test_trials, 1)), dim=0)

        self.noise_var = args.noise_var
        self.mld_err = 0

    def _sample_points(self, n_points):
        diff = np.linspace(0, 2*np.pi, self.N, endpoint=False).reshape((1, self.N)) -\
         np.random.uniform(0, 2*np.pi, size=(n_points, 1))

        _x1 = np.exp((np.cos(diff) - 1) / self.sig_s1**2)
        _x1 /= np.linalg.norm(_x1) / np.sqrt(self.N * n_points)
        _x2 = np.exp((np.cos(diff) - 1) / self.sig_s2**2)
        _x2 /= np.linalg.norm(_x2) / np.sqrt(self.N * n_points)
        rand_amplitude = np.random.uniform(0.5, 1.5, size=(n_points, 1))

        return np.concatenate((_x1 * rand_amplitude, _x2 * rand_amplitude), axis=0)

    def train_mean(self):
        return torch.from_numpy(self._sample_points(self.n_train_trials)).float()

    def test_mean(self):
        return torch.from_numpy(self._sample_points(self.n_test_trials)).float()



class Saver:
    """
    Saver v1.3. Created on 2/3/2019.
    A simple handler of a pickle-based database that stores data arrays.
    v1.1 020419: added remove; fixed a bug where the old data is always erased
    v1.2 032319: when saving, it will now check whether there's already an entry with the same tag
    v.13 032619: add_trials_to() allows appending existing results with new trials; shows num of data points
    v2 062119: add the "param" keyword, which saves specified parameters as a dictionary
    """

    def __init__(self, file_name, overwrite=False):
        if os.path.isfile(file_name) is True:
            print(f'Saver: File with name "{file_name}" already exists')

            if overwrite is False:
                _exist = pickle.load(open(file_name, 'rb'))
                self.data = _exist
        else:
            if overwrite:
                print('File to be overwritten does not exist.')
            self.data = []
        self.file_name = file_name

        if overwrite:
            if input('Overwrite existing file? [y/n]') == 'y':
                pickle.dump(self.data, open(file_name, 'wb'))
                print('overwritten.')
        else:
            pickle.dump(self.data, open(file_name, 'wb'))

    def save(self, x, y, tag, params=None, xl=None, yl=None):
        self.data = pickle.load(open(self.file_name, 'rb'))
        assert len(y.shape) == 2
        assert len(x) == y.shape[0]

        for _entry in self.data:
            if tag == _entry['tag']:
                raise ValueError('An entry with the given tag already exists.')
        self.data.append({'tag':tag, 'time':str(datetime.datetime.now()),
                          'x':x, 'y':y, 'p':params, 'xl':xl, 'yl':yl})
        pickle.dump(self.data, open(self.file_name, 'wb'))
        print('Saved as:', tag)

    def show(self):
        _ind = 0
        print('{:<3} | {:<40} | {:<30} | {:<5}'.format('ind', 'tag', 'time', '#data'))
        for _item in self.data:
            print('{:<3} | {:<40} | {:<30} | {:<5}'.format(_ind, _item['tag'], _item['time'], _item['y'].shape[1]))
            _ind += 1

    def remove(self, ind):
        if input('Remove entry #' + str(ind) + '?[y/n]') == 'y':
            self.data.pop(ind)
            pickle.dump(self.data, open(self.file_name, 'wb'))
            print('done')

    def add_trials_to(self, ind, new_data):
        if input(f'Add results to slot #{ind}? (should be in format of dim * trials)') == 'y':
            _old_data = self.data[ind]['y']
            _full_data = np.concatenate((_old_data, new_data), axis=1)
            self.data[ind]['y'] = _full_data
            pickle.dump(self.data, open(self.file_name, 'wb'))
            print('done')


class Reader(torch.nn.Module):
    def __init__(self, v_size, nlearn, eta0=0.001):

        super(Reader, self).__init__()
        self._eta0 = eta0
        self.reader = torch.nn.Linear(v_size, 1)
        self.v_size = v_size
        [self.V, self.b] = list(self.reader.parameters())
        self.optim = torch.optim.Adam([self.V, self.b], lr=self._eta0, eps=1e-4, amsgrad=True)
        # self.optim = torch.optim.SGD([self.V, self.b], lr=0.001, momentum=0.9)
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(self.optim, verbose=True, patience=8)
        self.init_scheduler_state = self.scheduler.state_dict()
        self.div_counter = 0

    def get_cross_ent_loss_and_update(self, train_trials):
        self.zero_grad()
        _n_train_trials = int(train_trials.shape[0] / 2)
        _r_arr = self.forward(train_trials)
        rp_arr = _r_arr[:_n_train_trials]
        rm_arr = _r_arr[_n_train_trials:]
        _loss = torch.sum(-torch.log(torch.sigmoid(rp_arr)) - torch.log(1 - torch.sigmoid(rm_arr)))
        _loss /= _n_train_trials
        _loss.backward()
        self.optim.step()

    def forward(self, x):
        return self.reader(x)

    def test(self, input_arr):
        with torch.no_grad():
            _n_test_trials = int(input_arr.shape[0] / 2)
            _r_arr = self.forward(input_arr).data.numpy()
            _err = (np.sum(_r_arr[:_n_test_trials] < 0) + np.sum(_r_arr[_n_test_trials:] > 0)) / (2 * _n_test_trials)
        return _err

    def check_div(self):
        _div = False
        if torch.sum(torch.isnan(self.V) + torch.isnan(self.b)) > 0:
            self.div_counter += 1
            if self.div_counter > 5:
                raise RuntimeError('Diverged for too many times in this trial.')
            # reset weights
            self.V.data = torch.normal(mean=torch.zeros_like(self.V.data)) / self.v_size
            self.b.data = torch.zeros(1)
            _div = True
            # reset the optimizer
            self.optim = torch.optim.Adam([self.V, self.b], lr=self._eta0 / 10, eps=1e-4)
            # reset the scheduler
            # self.scheduler.load_state_dict(self.init_scheduler_state)
        return _div
