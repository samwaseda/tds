import numpy as np
from pyiron_base.job.generic import GenericJob
from pyiron_base.generic.object import HasStorage
from pint import UnitRegistry
from tqdm import tqdm


class Diffusion(GenericJob, HasStorage):
    def __init__(self, project, job_name):
        GenericJob.__init__(self, project, job_name)
        HasStorage.__init__(self)
        self.storage.create_group("input")
        self.storage.create_group("output")
        self._python_only_job = True
        self.input.c_0 = 0.1
        self.input.c_min = 0
        self.input.spacing = 0.01
        self.input.length = 100
        self.input.gb_positions = np.array([0.25, 0.75])
        self.input.sigma = 0.5
        self.input.e_binding = 0.4
        self.input.D_0 = 0.352**2 * 1e13
        self.input.edge_slope = 1
        self.input.init_dt = 1e-9
        self.input.dt_up = 1.1
        self.input.dt_down = 0.5
        self.input.max_change = 0.01
        self.input.n_steps = 10000000
        self.input.dTdt = 10000000
        self.input.temperature = 300
        self.input.diffusion_barrier = 0.46
        self.input.n_snapshots = 10
        self._z = None
        self._epsilon = None
        self._d_epsilon = None
        self._dd_epsilon = None
        self._kB = None
        self._temperature = None
        self._c = None

    @property
    def gb_positions(self):
        return self.input.gb_positions * self.input.length

    @property
    def c(self):
        if self._c is None:
            cc = self.input.c_0 / (1 - self.input.c_0)
            exp = np.exp(-self.epsilon / self.kBT)
            c = cc * exp / (1 + cc * exp)
            self._c = self.get_slope(self.z) * self.get_slope(self.z.max() - self.z) * c
        return self._c

    @c.setter
    def c(self, c_new):
        self._c = c_new

    @property
    def temperature(self):
        if self._temperature is None:
            self._temperature = self.input.temperature
        return self._temperature

    @temperature.setter
    def temperature(self, T):
        self._temperature = T

    @property
    def kBT(self):
        return self.temperature * self.kB

    @property
    def kB(self):
        if self._kB is None:
            unit = UnitRegistry()
            self._kB = (1 * unit.boltzmann_constant * unit.kelvin).to('eV').magnitude
        return self._kB

    def get_slope(self, z):
        exp = np.exp(-self.input.edge_slope * z)
        return (1 - exp) / (1 + exp)

    def get_eps(self, x):
        return np.exp(-x**2 / (2 * self.input.sigma**2))

    @property
    def epsilon(self):
        if self._epsilon is None:
            self._epsilon = -self.input.e_binding * np.sum([
                self.get_eps(self.z - xx)
                for xx in self.gb_positions
            ], axis=0)
        return self._epsilon

    @property
    def d_epsilon(self):
        if self._d_epsilon is None:
            self._d_epsilon = self.input.e_binding * np.sum([
                (self.z - xx) / self.input.sigma**2 * self.get_eps(self.z - xx)
                for xx in self.gb_positions
            ], axis=0)
        return self._d_epsilon

    @property
    def dd_epsilon(self):
        if self._dd_epsilon is None:
            self._dd_epsilon = -self.input.e_binding * np.sum([
                ((self.z - xx)**2 / self.input.sigma**2 - 1) * self.get_eps(self.z - xx)
                for xx in self.gb_positions
            ], axis=0) / self.input.sigma**2
        return self._dd_epsilon

    @property
    def z(self):
        if self._z is None:
            self._z = np.linspace(0, self.input.length, self.n_mesh)
        return self._z

    @property
    def n_mesh(self):
        return int(self.input.length / self.input.spacing)

    @property
    def dc(self):
        return np.gradient(self.c, edge_order=2) / self.input.spacing

    @property
    def ddc(self):
        return (np.roll(self.c, 1) + np.roll(self.c, -1) - 2 * self.c) / self.input.spacing**2

    @property
    def diff_coeff(self):
        return self.input.D_0 * np.exp(-self.input.diffusion_barrier / self.kBT)

    @property
    def dcdt(self):
        return self.diff_coeff * self.ddc + self.diff_coeff / self.kBT * (
            self.dc * (1 - 2 * self.c) * self.d_epsilon + self.c * (1 - self.c) * self.dd_epsilon
        )

    def _initialize_output(self):
        self.output.h_lst = np.zeros(self.input.n_steps)
        self.output.T_lst = np.zeros(self.input.n_steps)
        self.output.t_lst = np.zeros(self.input.n_steps)
        self.output.c_lst = np.zeros((self.input.n_snapshots, len(self.c)))

    def run_diffusion(self):
        i_ss = np.rint(np.linspace(0, self.input.n_steps - 1, self.input.n_snapshots)).astype(int)
        dt = self.input.init_dt
        for ii in tqdm(range(self.input.n_steps)):
            dcdt = self.dcdt
            dt *= self.input.dt_up
            while np.any(self.c * self.input.max_change < -dcdt * dt):
                dt *= self.input.dt_down
            dc = dt * dcdt
            self.output.h_lst[ii] = dc[0] + dc[-1]
            self.output.T_lst[ii] = self.temperature
            self.output.t_lst[ii] = dt
            self.temperature += dt * self.input.dTdt
            self.c += dc
            self.c[0] = self.c[-1] = self.input.c_min
            if ii in i_ss:
                self.output.c_lst[np.where(ii == i_ss)[0]] = self.c

    def collect_output(self):
        pass

    def collect_logfiles(self):
        pass

    def run_static(self):
        self._initialize_output()
        self.run_diffusion()
        self.status.collect = True
        self.run()
        self.status.finished = True
        self.to_hdf()

    def _check_if_input_should_be_written(self):
        return False

    @property
    def input(self):
        return self.storage.input

    @property
    def output(self):
        return self.storage.output

    def to_hdf(self, hdf=None, group_name=None):
        GenericJob.to_hdf(self, hdf=hdf, group_name=group_name)
        HasStorage.to_hdf(self, hdf=self.project_hdf5, group_name="")

    def from_hdf(self, hdf=None, group_name=None):
        GenericJob.from_hdf(self, hdf=hdf, group_name=group_name)
        HasStorage.from_hdf(self, hdf=self.project_hdf5, group_name="")
