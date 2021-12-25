import numpy as np
from pyiron_base.generic.datacontainer import DataContainer
from pyiron_atomistics.atomistics.job.interactivewrapper import InteractiveWrapper


class Metadynamics(InteractiveWrapper):
    def __init__(self, project, job_name):
        super().__init__(project, job_name)
        self.input = DataContainer(table_name='input')
        self.output = DataContainer(table_name='output')
        self.input.update_every_n_steps = 100
        self.input.sigma = 0.0001
        self.input.spacing = 0.25
        self.input.increment = 0.001
        self.input.E_min = None
        self.input.E_max = None
        self._mesh = None
        self._index_H = None
        self._index_Ni = None

    @property
    def sigma(self):
        return self.input.sigma * len(self.structure)

    @property
    def spacing(self):
        return self.sigma * self.input.spacing

    @property
    def mesh(self):
        if self._mesh is None:
            self._mesh = np.arange(self.input.E_min, self.input.E_max, self.spacing)
        return self._mesh

    def validate_ready_to_run(self):
        super().validate_ready_to_run()
        if self.input.E_min is None or self.input.E_max is None:
            raise ValueError('E_min and/or E_max not set')

    @property
    def index_H(self):
        if self._index_H is None:
            self._index_H = self.ref_job.structure.select_index('H')[0]
        return self._index_H

    @property
    def index_Ni(self):
        if self._index_Ni is None:
            self._index_Ni = self.ref_job.structure.select_index('Ni')
        return self._index_Ni

    def run_static(self):
        self.output.B = np.zeros(len(self.mesh))
        self.output.dBds = np.zeros(len(self.mesh))
        self.status.running = True
        self.ref_job_initialize()
        self.ref_job.input.control['thermo'] = '1'
        self.ref_job._log_file = 'none'
        self.ref_job.set_fix_external(self.callback, overload_internal_fix_external=True)
        self.ref_job.run()
        self.status.collect = True
        self.ref_job.interactive_close()
        self.status.finished = True
        self.to_hdf()

    def callback(self, caller, ntimestep, nlocal, tag, x, fext):
        E = self.ref_job.interactive_energy_pot_getter()
        tags = tag.flatten().argsort()
        fext.fill(0)
        f = self.get_force(E)
        fext[tags[self.index_H]] += f
        fext[tags[self.index_Ni]] -= f / (len(tag) - 1)
        if ((ntimestep + 1) % self.input.update_every_n_steps) == 0:
            self.update_s(E)

    def get_force(self, E):
        index = int((E - self.input.E_min) / self.spacing)
        if index < 0 or index >= len(self.mesh):
            return 0
        f = self.ref_job.interactive_forces_getter()
        return self.output.dBds[index] * np.asarray(f[self.index_H])

    def update_s(self, E):
        dE = self.mesh[:, None] - E
        exp = np.exp(-dE**2 / 2 / self.sigma**2)
        self.output.dBds -= self.input.increment / self.sigma**2 * np.sum(dE * exp, axis=1)
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
