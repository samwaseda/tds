import numpy as np
from pyiron_base.generic.datacontainer import DataContainer
from pyiron_atomistics.atomistics.job.interactivewrapper import InteractiveWrapper


class Metadynamics(InteractiveWrapper):
    def __init__(self, project, job_name):
        super().__init__(project, job_name)
        self.input = DataContainer(table_name='input')
        self.output = DataContainer(table_name='output')
        self.input.n_print = 1000
        self.input.update_every_n_steps = 100
        self.input.sigma = 0.2
        self.input.spacing = 0.25
        self.input.increment = 0.0001
        self.input.symprec = 1.0e-2
        self.input.unit_length = 0
        self.input.axis = None
        self._symmetry = None
        self._mesh = None

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
        return self._mesh

    @property
    def symmetry(self):
        if self._symmetry is None:
            x_random = np.random.random(3)
            sym = self.structure_unary.get_symmetry()
            x_sym = self.structure_unary.get_wrapped_coordinates(
                np.einsum('nij,j->ni', sym.rotations, x_random) + sym.translations
            )
            sym_indices = np.unique(x_sym[:, self.input.axis], return_index=True)[1]
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

    def run_static(self):
        self.output.B = np.zeros(len(self.mesh))
        self.output.dBds = np.zeros(len(self.mesh))
        self.status.running = True
        self.ref_job_initialize()
        self.ref_job.set_fix_external(self.callback, overload_internal_fix_external=True)
        self.ref_job.run()
        self.status.collect = True
        self.ref_job.interactive_close()
        self.status.finished = True
        self.to_hdf()

    def callback(self, caller, ntimestep, nlocal, tag, x, fext):
        tags = tag.flatten().argsort()
        fext.fill(0)
        f = self.get_force(x[tags[-1], self.input.axis])
        fext[tags[-1], self.input.axis] += f
        fext[tags[:-1], self.input.axis] -= f / (len(tag) - 1)
        if ((ntimestep + 1) % self.input.update_every_n_steps) == 0:
            self.update_s(x[tags[-1], self.input.axis])

    def get_force(self, x):
        index = int(x / self.spacing) % len(self.mesh)
        return -self.output.dBds[index]

    def _get_symmetric_x(self, x):
        x_scaled = x / self.structure.cell[self.input.axis, self.input.axis]
        x_new = self.symmetry['rotations'] * x_scaled + self.symmetry['translations']
        x_new -= np.floor(x_new)
        x_repeated = (x_new[:, None] + np.array([-1, 0, 1])).flatten()
        return self.structure.cell[self.input.axis, self.input.axis] * x_repeated

    def update_s(self, x):
        x = self._get_symmetric_x(x)
        dx = self.mesh[:, None] - x
        exp = np.exp(-dx**2 / 2 / self.input.sigma**2)
        self.output.dBds -= self.input.increment / self.input.sigma**2 * np.sum(dx * exp, axis=1)
        self.output.B += self.input.increment * np.sum(exp, axis=1)

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
