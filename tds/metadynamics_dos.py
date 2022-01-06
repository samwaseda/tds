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
        self.input.spacing = 0.2
        self.input.increment = 0.001
        self.input.E_min = None
        self.input.E_max = None
        self.input.use_derivative = True
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
            self._mesh += 0.5 * self.spacing
        return self._mesh

    def validate_ready_to_run(self):
        super().validate_ready_to_run()
        if self.input.E_min is None or self.input.E_max is None:
            raise ValueError('E_min and/or E_max not set')

    def run_static(self):
        self.output.B = np.zeros(len(self.mesh))
        self.output.dBds = np.zeros(len(self.mesh))
        self.output.ddBdds = np.zeros(len(self.mesh))
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
        E = self.ref_job.interactive_energy_tot_getter()
        tags = tag.flatten().argsort()
        fext.fill(0)
        f = self.get_force(E)
        fext[tags] += f - np.mean(f, axis=0)
        if ((ntimestep + 1) % self.input.update_every_n_steps) == 0:
            self.update_s(E)

    def interactive_velocities_getter(self):
        return np.reshape(
            np.array(self.ref_job._interactive_library.gather_atoms("v", 1, 3)),
            (len(self.structure), 3),
        )

    def get_force(self, E):
        index = np.rint((E - self.input.E_min) / self.spacing).astype(int)
        f = self.ref_job.interactive_forces_getter()
        if index >= len(self.mesh):
            v = np.asarray(self.interactive_velocities_getter())
            return -np.einsum('i,i,ij->ij', np.linalg.norm(f), 1 / np.linalg.norm(v), v)
        elif index < 0:
            return np.random.randn(*self.structure.positions.shape)
        dBds = self.output.dBds[index]
        if self.input.use_derivative:
            dBds += self.output.ddBdds[index] * (E - self.mesh[index])
        return  dBds * np.asarray(f)

    def update_s(self, E):
        dE_rel = (self.mesh[:, None] - E) / self.sigma
        exp = np.exp(-dE_rel**2 / 2)
        self.output.dBds -= self.input.increment / self.sigma * np.sum(dE_rel * exp, axis=1)
        self.output.B += self.input.increment * np.sum(exp, axis=1)
        if self.input.use_derivative:
            self.output.ddBdds += self.input.increment / self.sigma**2 * np.sum(
                (dE_rel**2 - 1) * exp, axis=1
            )

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
