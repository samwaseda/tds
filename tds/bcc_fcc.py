import numpy as np
from pyiron_base.job.generic import GenericJob
from pyiron_base.generic.object import HasStorage
from pint import UnitRegistry
from collections import defaultdict
from tqdm import tqdm


class Diffusion(GenericJob, HasStorage):
    def __init__(self, project, job_name):
        GenericJob.__init__(self, project, job_name)
        HasStorage.__init__(self)
        self.storage.create_group("input")
        self.storage.create_group("output")
        self._python_only_job = True
        self.input.length = 10
        self.input.bccfcc = 0.5 * self.input.length
        self.input.D_0 = 1.0e12
        self.input.width = 1
        self.input.E_mis = -0.2
        self.input.E_bcc = 0.08
        self.input.Q_fcc = 0.44
        self.input.Q_bcc = 0.05
        self.input.velocity = 1
        self.input.e_diff_bcc_fcc = 0.001
        self.input.spacing = 0.01
        self.input.temperature = 1000
        self.input.density = 4 / 3.6**3
        self._ureg = UnitRegistry()
        self._x = None
        self.kB = (1 * self._ureg.kelvin * self._ureg.boltzmann_constant).to('eV').magnitude
        self.input.c_0 = 0.01
        self._freq = None
        self._f_dict = defaultdict(lambda: None)
        self._E_dict = defaultdict(lambda: None)
        self._Q_dict = defaultdict(lambda: None)
        self._D_dict = defaultdict(lambda: None)
        self._c = None
        self.input.init_dt = 1.0e-9
        self.input.dt_up = 0.1
        self.input.dt_down = 0.5
        self.input.n_steps = 10000
        self.input.n_snapshots = 100
        self.input.max_change = 0.01
        self.t_tot = 0
        self._i_output = None
        self._bccfcc = None

    @property
    def c(self):
        if self._c is None:
            self._c = self.input.c_0 * np.ones(self.n_mesh)
        return self._c

    def reset(self):
        self._f_dict = defaultdict(lambda: None)
        self._E_dict = defaultdict(lambda: None)
        self._Q_dict = defaultdict(lambda: None)
        self._D_dict = defaultdict(lambda: None)

    def set_concentration(self, chemical_potential=None, c_0=None, temperature=None):
        if chemical_potential is None and c_0 is None:
            print('Neither chemical potential nor c_0 given')
            return
        kBT = self.kBT
        if temperature is not None:
            kBT = self.kB * temperature
        if chemical_potential is not None:
            self._c = 1 / (1 + np.exp((self.get_E() - chemical_potential) / kBT))
        else:
            self._c = self.ones_like(self.c) * c_0

    @property
    def bccfcc(self):
        if self._bccfcc is None:
            self._bccfcc = self.input.bccfcc
        return self._bccfcc

    @bccfcc.setter
    def bccfcc(self, value):
        self.reset()
        self._bccfcc = value

    @property
    def freq(self):
        if self._freq is None:
            self._freq = 2 * np.pi * 1j * np.fft.fftfreq(len(self.x), self.input.spacing)
        return self._freq

    def _get_gradient(self, x):
        return np.fft.ifft(np.fft.fft(x) * self.freq).real

    def _get_laplace(self, x):
        return np.fft.ifft(np.fft.fft(x) * self.freq**2).real

    @property
    def n_mesh(self):
        return int(2 * self.input.length / self.input.spacing)

    @property
    def kBT(self):
        return self.kB * self.input.temperature

    @property
    def x(self):
        if self._x is None:
            self._x = np.linspace(
                -self.input.length, self.input.length, self.n_mesh, endpoint=False
            )
        return self._x

    @property
    def sigma(self):
        return 0.5 * self.input.width / np.log(10)

    def get_f(self, order=0, absolute=False):
        key = '{}_{}'.format(order, absolute)
        if self._f_dict[key] is not None:
            return self._f_dict[key]
        f = 1 / (1 + np.exp(-((self.x + np.array([[1], [-1]]) * self.bccfcc) / self.sigma)))
        f = self._get_f_deriv(f, order=order)
        if absolute:
            f = np.sum(f, axis=0)
        else:
            f = -np.diff(f, axis=0).squeeze()
        self._f_dict[key] = f
        return self._f_dict[key]

    def _get_f_deriv(self, f, order=0):
        if order == 0:
            return f
        if order == 1:
            return f * (1 - f) / self.sigma
        elif order == 2:
            return f * (1 - f) * (1 - 2 * f) / self.sigma**2
        elif order == 3:
            return f * (1 - f) * (1 - 6 * f + 6 * f**2) / self.sigma**3

    def get_Q(self, order=0):
        if self._Q_dict[order] is not None:
            return self._Q_dict[order]
        self._Q_dict[order] = (self.input.Q_bcc - self.input.Q_fcc) * self.get_f(order=order)
        if order == 0:
            self._Q_dict[order] += self.input.Q_fcc
        return self._Q_dict[order]

    def get_D(self, order=0):
        if self._D_dict[order] is not None:
            return self._D_dict[order]
        self._D_dict[order] = self.input.D_0 * np.exp(-self.get_Q(order=0) / self.kBT)
        if order == 1:
            self._D_dict[order] *= -self.get_Q(order=1) / self.kBT
        return self._D_dict[order]

    def get_E(self, order=0, kBT=False):
        if self._E_dict[order] is not None:
            if kBT:
                return self._E_dict[order] / self.kBT
            else:
                return self._E_dict[order]
        self._E_dict[order] = 4 * self.sigma * (
            self.input.E_mis - 0.5 * self.input.E_bcc
        ) * self.get_f(order=order + 1, absolute=True)
        self._E_dict[order] += self.input.E_bcc * self.get_f(order=order, absolute=False)
        return self.get_E(order=order, kBT=kBT)

    @property
    def dcdx(self):
        return self._get_gradient(self.c)

    @property
    def ddcddx(self):
        return self._get_laplace(self.c)

    @property
    def dcdt(self):
        dDdx_first = (self.c * (1 - self.c) * self.get_E(order=1, kBT=True))
        dDdx_second = self.dcdx
        dDdx = self.get_D(order=1) * (dDdx_first + dDdx_second)
        D_first = (1 - 2 * self.c) * self.dcdx * self.get_E(order=1, kBT=True)
        D_second = self.c * (1 - self.c) * self.get_E(order=2, kBT=True)
        D = (D_first + D_second + self.ddcddx) * self.get_D()
        return dDdx + D

    def _get_dc(self, dcdt, dt):
        c_dc = dcdt.sum() / self.c.sum()
        return dt * (dcdt - self.c * c_dc) / (1 + dt * c_dc)

    @property
    def i_output(self):
        if self._i_output is None:
            self._i_output = np.rint(
                np.linspace(0, self.input.n_steps - 1, self.input.n_snapshots)
            ).astype(int)
        return self._i_output

    def _check_dt(self, dc):
        if np.any(self.c[1:-1] * self.input.max_change < -dc[1:-1]):
            return True
        elif np.any(dc[1:-1] > self.input.max_change * (1 - self.c[1:-1])):
            return True
        return False

    def _initialize_output(self):
        self.output.free_energy = np.zeros(self.input.n_snapshots)
        self.output.energy = np.zeros(self.input.n_snapshots)
        self.output.time = np.zeros(self.input.n_snapshots)
        self.output.distribution = np.zeros((self.input.n_snapshots, len(self.c)))
        self.output.interface_position = np.zeros(self.input.n_snapshots)

    def run_static(self):
        self._initialize_output()
        self.status.running = True
        self.run_diffusion()
        self.status.collect = True
        self.run()
        self.status.finished = True
        self.to_hdf()

    @property
    def ext_force(self):
        if self.input.e_diff_bcc_fcc is None:
            return None
        return self.input.density * self.input.e_diff_bcc_fcc

    def _get_next_position(self, dt):
        if self.ext_force is not None:
            dfdx = self._dfdx
            if np.isclose(dfdx, 0):
                return self.input.velocity * dt * (self.ext_force + self.force)
            return (self.ext_force + self.force) / dfdx * (
                np.exp(self.input.velocity * dfdx * dt) - 1
            )
        return self.input.velocity * dt

    def run_diffusion(self):
        dt = np.log(self.input.init_dt)
        counter = 0
        for ii in tqdm(range(self.input.n_steps)):
            dcdt = self.dcdt
            dt += self.input.dt_up
            while self._check_dt(self._get_dc(dcdt, np.exp(dt))):
                dt -= self.input.dt_down
            dc = self._get_dc(dcdt, np.exp(dt))
            self.t_tot += np.exp(dt)
            self._c += dc
            self.bccfcc += self._get_next_position(np.exp(dt))
            if ii == self.i_output[counter]:
                self.output.free_energy[counter] = np.mean(self.free_energy) * self._measure
                self.output.energy[counter] = np.mean(self.energy) * self._measure
                self.output.distribution[counter] = self.c
                self.output.time[counter] = self.t_tot
                self.output.interface_position[counter] = self.bccfcc
                counter += 1
            if self.bccfcc > self.input.length:
                break

    @property
    def entropy(self):
        return self.kBT * (self.c * np.log(self.c) + (1 - self.c) * np.log(1 - self.c))

    @property
    def energy(self):
        return self.c * self.get_E()

    @property
    def _measure(self):
        return self.input.length * self.input.density * 2

    @property
    def _dfdx(self):
        force = 4 * self.sigma * (
            self.input.E_mis - 0.5 * self.input.E_bcc
        ) * self.get_f(order=3, absolute=True)
        force += self.input.E_bcc * self.get_f(order=2, absolute=False)
        return -np.mean(force * self.c) * self._measure

    @property
    def force(self):
        force = 4 * self.sigma * (
            self.input.E_mis - 0.5 * self.input.E_bcc
        ) * self.get_f(order=2, absolute=False)
        force += self.input.E_bcc * self.get_f(order=1, absolute=True)
        return -np.mean(force * self.c) * self._measure

    @property
    def chemical_potential(self):
        return self.get_E() + self.kBT * np.log(self.c / (1 - self.c))

    @property
    def free_energy(self):
        return self.energy + self.entropy

    def collect_output(self):
        pass

    def collect_logfiles(self):
        pass

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
