import numpy as np
from pyiron_base.generic.datacontainer import DataContainer
from pyiron_atomistics.atomistics.job.interactivewrapper import InteractiveWrapper


class Metadynamics(InteractiveWrapper):
    def __init__(self, project, job_name):
        super().__init__(project, job_name)
        self.input = DataContainer(table_name='input')
        self.output = DataContainer(table_name='output')
        self.input.update_every_n_steps = 100
        self.input.sigma = 0.2
        self.input.spacing = 0.25
        self.input.increment = 0.0001
        self.input.symprec = 1.0e-1
        self.input.axis = None
        self.input.decay = 1.0
        self.input.use_gradient = True
        self._symmetry = None
        self._mesh = None
        self._current_increment = None
        self._ind_meta = None
        self._ind_nonmeta = None

    @property
    def spacing(self):
        return self.input.sigma * self.input.spacing

    @property
    def length(self):
        return self.structure.cell[self.input.axis, self.input.axis]

    @property
    def mesh(self):
        if self._mesh is None:
            self._mesh = np.arange(0, self.length, self.spacing)
            self._mesh += 0.5 * self.spacing
        return self._mesh

    @property
    def symmetry(self):
        if self._symmetry is None:
            x_random = np.random.random(3)
            sym = self.structure_unary.get_symmetry(symprec=self.input.symprec)
            x_sym = self.structure_unary.get_wrapped_coordinates(
                np.einsum('nij,j->ni', sym.rotations, x_random) + sym.translations
            )
            sym_indices = np.unique(
                np.round(x_sym[:, self.input.axis], decimals=8), return_index=True
            )[1]
            self._symmetry = {
                'rotations': sym.rotations[sym_indices, self.input.axis, self.input.axis],
                'translations': sym.translations[sym_indices, self.input.axis]
            }
        return self._symmetry

    def validate_ready_to_run(self):
        super().validate_ready_to_run()
        if self.input.axis is None:
            raise ValueError('Axis not set')

    @property
    def structure_unary(self):
        return self.structure[
            self.structure.select_index(self.structure.get_majority_species()['symbol'])
        ]

    def _initialize_potentials(self):
        self._current_increment = self.input.increment
        self.output.B = np.zeros(len(self.mesh))
        self.output.dBds = np.zeros(len(self.mesh))
        self.output.ddBdds = np.zeros(len(self.mesh))

    def run_static(self):
        self._initialize_potentials()
        self.status.running = True
        self.ref_job_initialize()
        self.ref_job.set_fix_external(self.callback, overload_internal_fix_external=True)
        self.ref_job.run()
        self.status.collect = True
        self.ref_job.interactive_close()
        self.status.finished = True
        self.to_hdf()

    @property
    def ind_nonmeta(self):
        if self._ind_nonmeta is None:
            self._ind_nonmeta = self.structure.select_index('Ni')
        return self._ind_nonmeta

    @property
    def ind_meta(self):
        if self._ind_meta is None:
            self._ind_meta = self.structure.select_index('H')
        return self._ind_meta

    def callback(self, caller, ntimestep, nlocal, tag, x, fext):
        tags = tag.flatten().argsort()
        fext.fill(0)
        f = self.get_force(x[tags[self.ind_meta], self.input.axis])
        fext[tags[self.ind_meta], self.input.axis] += f
        fext[tags[self.ind_nonmeta], self.input.axis] -= f.mean(axis=0) / len(self.ind_nonmeta)
        if ((ntimestep + 1) % self.input.update_every_n_steps) == 0:
            self.update_s(x[tags[self.ind_meta], self.input.axis])

    def get_force(self, x):
        index = np.rint(x / self.spacing).astype(int) % len(self.mesh)
        dBds = self.output.dBds[index]
        if self.input.use_gradient:
            dx = x - self.mesh[index]
            dx -= self.length * np.rint(dx / self.length)
            dBds += dx * self.output.ddBdds[index]
        return -dBds

    def _get_symmetric_x(self, x):
        x_scaled = x / self.structure.cell[self.input.axis, self.input.axis]
        x_new = np.einsum(
            '...,j->...j', x_scaled, self.symmetry['rotations']
        ) + self.symmetry['translations']
        x_new -= np.floor(x_new)
        x_repeated = (x_new.reshape(-1, 1) + np.array([-1, 0, 1])).flatten()
        return self.structure.cell[self.input.axis, self.input.axis] * x_repeated

    def update_s(self, x):
        x = self._get_symmetric_x(x)
        dx = self.mesh[:, None] - x
        dx -= self.length * np.rint(dx / self.length)
        dx /= self.input.sigma
        exp = np.exp(-dx**2 / 2)
        self.output.dBds -= self._current_increment / self.input.sigma * np.sum(dx * exp, axis=1)
        self.output.B += self._current_increment * np.sum(exp, axis=1)
        if self.input.use_gradient:
            self.output.ddBdds += self._current_increment / self.input.sigma**2 * np.sum(
                (dx**2 - 1) * exp, axis=1
            )
        self._current_increment *= self.input.decay

    def to_hdf(self, hdf=None, group_name=None):
        super().to_hdf(
            hdf=hdf,
            group_name=group_name
        )
        self.output.to_hdf(hdf=self.project_hdf5, group_name='output')

    def from_hdf(self, hdf=None, group_name=None):
        super().from_hdf(
            hdf=hdf,
            group_name=group_name
        )
        self.output.from_hdf(hdf=self.project_hdf5, group_name='output')

    def write_input(self):
        pass

    def get_energy(self, x):
        index = int(x / self.spacing) % len(self.mesh)
        return self.output.B[index]

    @property
    def filling_rate(self):
        dEV = (np.sqrt(np.pi) * self.input.sigma) * self.input.increment
        V_tot = self.length
        N_sym = len(self.symmetry['rotations'])
        N_freq = self.input.update_every_n_steps
        return dEV / V_tot * N_sym / N_freq
