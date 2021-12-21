import numpy as np
from pyiron_atomistics.atomistics.job.interactivewrapper import InteractiveWrapper
from pyiron_base import DataContainer


class ART(InteractiveWrapper):
    def __init__(self, project, job_name):
        super().__init__(project, job_name)
        self.__name__ = "ARTNouveau"
        self._ref_job = None
        self.input = Input()
        self.output = DataContainer(table_name='output')
        self._dx = None
        self.X = np.zeros((3, 3))
        self.Y = np.zeros((3, 3))
        self._non_diff_id = None
        self._hessian = None
        self._eigenvalues = None
        self._eigenvectors = None
        self.output.eigenvalues = []
        self.output.eigenvectors = []
        self.output.step_size = []
        self.current_force = None

    @property
    def non_diff_id(self):
        if self._non_diff_id:
            arr = np.arange(len(self.structure))
            self._non_diff_id = arr[arr != self.input.diff_id]
        return self._non_diff_id

    def write_input(self):
        pass

    def validate_ready_to_run(self):
        super().validate_ready_to_run()
        if self.input.diff_id is None:
            raise ValueError('Diffusion id not set')

    def reset(self):
        self._dx = None
        self.f_lst = np.zeros((self.input.number_of_displacements, len(self.structure), 3))
        self._hessian = None
        self._eigenvalues = None
        self._eigenvectors = None

    @property
    def dx(self):
        if self._dx is None:
            self._dx = np.random.randn(self.input.number_of_displacements, 3) * self.dx_magnitude
        return self._dx

    def update_data(self):
        self.X *= self.input.decay
        self.Y *= self.input.decay
        self.X += np.einsum('nj,nk->jk', self.dx, self.dx)
        self.Y += np.einsum(
            'ni,nk->ik',
            self.current_force[self.input.diff_id] - self.f_lst[:, self.input.diff_id], self.dx
        )

    @property
    def hessian(self):
        if self._hessian is None:
            self.update_data()
            self._hessian = np.einsum('kj,ik->ij', np.linalg.inv(self.X), self.Y)
            self._hessian = 0.5 * (self._hessian + self._hessian.T)
        return self._hessian

    @property
    def eigenvalues(self):
        if self._eigenvalues is None:
            self._eigenvalues, self._eigenvectors = np.linalg.eigh(self.hessian)
        return self._eigenvalues

    @property
    def eigenvectors(self):
        if self._eigenvectors is None:
            self._eigenvalues, self._eigenvectors = np.linalg.eigh(self.hessian)
        return self._eigenvectors

    @property
    def gamma(self):
        return self.input.gamma_prefactor * np.exp(
            -self.input.gamma_decay * self.eigenvalues[0]
        )

    @property
    def displacement(self):
        displacement = np.zeros_like(self.current_force)
        f = self.current_force.copy()
        if self.eigenvalues[0] < 0 and np.linalg.norm(
            f[self.input.diff_id]
        ) < self.input.harmonic_force:
            displacement[self.input.diff_id] = np.einsum(
                'ij,j->i', np.linalg.inv(self.hessian), f[self.input.diff_id]
            )
        else:
            df = -self.gamma * f[self.input.diff_id].dot(
                self.eigenvectors.T[0]
            ) * self.eigenvectors.T[0]
            displacement[self.input.diff_id] = df * self.input.step_size
        f[self.non_diff_id] -= np.mean(f[self.non_diff_id], axis=0)
        displacement[self.non_diff_id] = f[self.non_diff_id] * self.input.step_size
        return displacement

    def run_static(self):
        self.ref_job_initialize()
        f_prev = None
        for _ in range(self.ionic_steps):
            self.reset()
            self.ref_job.structure = self.structure.copy()
            self.ref_job.run()
            self.current_force = np.asarray(self.ref_job.interactive_forces_getter())
            if np.linalg.norm(self.current_force, axis=-1).max() < self.input.ionic_force_tolerance:
                break
            for i, x in enumerate(self.dx):
                self.ref_job.structure = self.structure.copy()
                self.ref_job.structure.positions[self.input.diff_id] += x
                self.ref_job.run()
                self.f_lst[i] = self.ref_job.interactive_forces_getter()
            self.structure.positions += self.displacement
            if f_prev is not None:
                if np.sum(self.current_force[:-1] * f_prev[:-1]) > 0:
                    self.input.step_size *= 1.1
                else:
                    self.input.step_size *= 0.5
            f_prev = self.current_force

    def collect_output(self):
        super().collect_output()
        self.output.eigenvalues.append(self.eigenvalues)
        self.output.eigenvectors.append(self.eigenvectors)
        self.output.step_size.append(self.input.step_size)

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


class Input(DataContainer):
    """
    Args:
        minimizer (str): minimizer to use (currently only 'CG' and 'BFGS' run
            reliably)
        ionic_steps (int): max number of steps
        ionic_force_tolerance (float): maximum force tolerance
    """

    def __init__(self, input_file_name=None, table_name="input"):
        self.ionic_steps = 100
        self.number_of_displacements = 5
        self.decay = 0.9
        self.ionic_force_tolerance = 1.0e-2
        self.harmonic_force = 1.0e-2
        self.step_size = 0.1
        self.diff_id = None
        self.dx_magnitude = 1e-3
        self.gamma_prefactor = 1
        self.gamma_decay = 1
