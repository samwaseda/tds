from scipy.spatial import cKDTree
import numpy as np
from pyiron_base.generic.datacontainer import DataContainer
from pyiron_atomistics.atomistics.job.interactivewrapper import InteractiveWrapper


class UnitCell:
    def __init__(
        self,
        unit_cell,
        sigma,
        increment,
        spacing=0.05,
        cutoff=4,
        symprec=1.0e-2,
        sigma_decay=1,
        increment_decay=1,
        min_sigma=0,
        min_increment=0,
        use_gradient=True,
    ):
        self.unit_cell = unit_cell
        self._sigma = sigma
        self._increment = increment
        self.spacing = spacing
        self._mesh = None
        self._tree_mesh = None
        self._symmetry = None
        self._x_repeat = None
        self._cutoff = cutoff
        self._x_lst = []
        self._cell_inv = None
        self.B = np.zeros(self.mesh.shape[:-1])
        self.dBds = np.zeros_like(self.mesh)
        self.ddBdds = np.zeros(self.mesh.shape + (3,))
        self._symmetry = None
        self._symprec = symprec
        self._gaussian_process = None
        self._sigma_decay = sigma_decay
        self._increment_decay = increment_decay
        self._min_sigma = min_sigma
        self._min_increment = min_increment
        self.use_gradient = use_gradient

    @property
    def sigma(self):
        return (
            self._sigma - self._min_sigma
        ) * self._sigma_decay ** len(self._x_lst) + self._min_sigma

    @property
    def increment(self):
        return (
            self._increment - self._min_increment
        ) * self._increment_decay ** len(self._x_lst) + self._min_increment

    @property
    def cutoff(self):
        return self._cutoff * self.sigma

    @property
    def num_neighbors(self):
        return int(1.5 * 4 / 3 * np.pi * self.cutoff**3 / self.spacing**3)

    def x_to_s(self, x):
        return self.unit_cell.get_wrapped_coordinates(x)

    @property
    def cell(self):
        return np.linalg.norm(self.unit_cell.cell, axis=-1)

    @property
    def mesh(self):
        if self._mesh is None:
            linspace = []
            for c in self.cell:
                ll = np.linspace(0, 1, np.rint(c / self.spacing).astype(int), endpoint=False)
                ll += 0.5 * (ll[1] - ll[0])
                linspace.append(ll)
            self._mesh = np.einsum(
                'j...,ji->...i',
                np.meshgrid(*linspace, indexing='ij'),
                self.unit_cell.cell
            )
        return self._mesh

    @property
    def tree_mesh(self):
        if self._tree_mesh is None:
            self._tree_mesh = cKDTree(self.mesh.reshape(-1, 3))
        return self._tree_mesh

    @property
    def symmetry(self):
        if self._symmetry is None:
            self._symmetry = self.unit_cell.get_symmetry(symprec=self._symprec)
        return self._symmetry

    def _get_symmetric_x(self, x_in, cutoff=None):
        if cutoff is None:
            cutoff = self.cutoff
        x = self.x_to_s(x_in)
        x = self.symmetry.generate_equivalent_points(x, return_unique=False)
        x = self.unit_cell.get_extended_positions(cutoff, positions=x)
        return x[self.tree_mesh.query(x)[0] < cutoff]

    def _get_neighbors(self, x):
        dist, indices = self.tree_mesh.query(
            x, k=self.num_neighbors, distance_upper_bound=self.cutoff
        )
        cond = dist < np.inf
        indices = indices[cond]
        dx = self.mesh.reshape(-1, 3)[indices] - x[np.indices(cond.shape)[0][cond]]
        return dist[cond], dx, np.unravel_index(indices, self.mesh.shape[:-1])

    def append_positions(self, x, symmetrize=True):
        if symmetrize:
            x = self._get_symmetric_x(x)
        dist, dx, unraveled_indices = self._get_neighbors(x)
        dx /= self.sigma
        B = self.increment * np.exp(-dist**2 / (2 * self.sigma**2))
        np.add.at(self.B, unraveled_indices, B)
        np.add.at(self.dBds, unraveled_indices, np.einsum('...i,...->...i', dx, B / self.sigma))
        if self.use_gradient:
            xx = (np.einsum('...i,...j->...ij', dx, dx) - np.eye(3)) / self.sigma**2
            np.add.at(self.ddBdds, unraveled_indices, np.einsum('...ij,...->...ij', xx, B))
        self._x_lst.extend(x)

    def _get_index(self, x):
        return np.unravel_index(
            self.tree_mesh.query(self.x_to_s(x))[1], self.mesh.shape[:-1]
        )

    def get_force(self, x):
        index = self._get_index(x)
        dBds = self.dBds[index].copy()
        if self.use_gradient:
            dx = x - self.mesh[index]
            dBds += np.einsum('...j,ij->...i', dx, self.ddBdds[index])
        return dBds

    @property
    def x_lst(self):
        return np.array(self._x_lst)

    @property
    def tree_output(self):
        return cKDTree(self.x_lst)

    @property
    def _num_neighbors_x_lst(self):
        rho = len(self.x_lst) / self.unit_cell.get_volume()
        return np.max([int(1.5 * 4 / 3 * np.pi * self.cutoff**3 * rho), 20])

    def get_energy(self, x, derivative=0):
        if derivative == 0:
            return -self.B[self._get_index(x)]
        elif derivative == 1:
            return -self.dBds[self._get_index(x)]
        elif derivative == 2:
            return -self.ddBdds[self._get_index(x)]
        else:
            raise ValueError(f"derivative = {derivative} does not exist")


class Metadynamics(InteractiveWrapper):
    def __init__(self, project, job_name):
        super().__init__(project, job_name)
        self.input = DataContainer(table_name='input')
        self.output = DataContainer(table_name='output')
        self.input.update_every_n_steps = 100
        self.input.increment = 0.001
        self.input.sigma = 0.2
        self.input.cutoff = 4
        self.input.sigma_decay = 1
        self.input.increment_decay = 1
        self.input.spacing = 0.05
        self.input.symprec = 1.0e-2
        self.input.unit_length = 0
        self.input.min_sigma = 0
        self.input.min_increment = 0
        self.input.use_gradient = True
        self.input.track_vacancy = False
        self._unit_cell = None
        self.output.x = []
        self._tree = None
        self._mass_ratios = None
        self.total_displacements = None
        self.x_previous = None

    @property
    def structure_unary(self):
        return self.structure[
            self.structure.select_index(self.structure.get_majority_species()['symbol'])
        ]

    @property
    def primitive_cell(self):
        return self.structure_unary.get_symmetry().get_primitive_cell()

    @property
    def unit_cell(self):
        if self._unit_cell is None:
            self._unit_cell = UnitCell(
                unit_cell=self.primitive_cell,
                sigma=self.input.sigma,
                increment=self.input.increment,
                spacing=self.input.spacing,
                cutoff=self.input.cutoff,
                symprec=self.input.symprec,
                sigma_decay=self.input.sigma_decay,
                increment_decay=self.input.increment_decay,
                min_sigma=self.input.min_sigma,
                min_increment=self.input.min_increment,
                use_gradient=self.input.use_gradient,
            )
        return self._unit_cell

    @property
    def mass_ratios(self):
        if self._mass_ratios is None:
            self._mass_ratios = self.structure.get_masses()
            self._mass_ratios /= self._mass_ratios.sum()
        return self._mass_ratios

    def collect_output(self):
        self.output.x = self.unit_cell.x_lst
        self.output.B = self.unit_cell.B
        self.output.dBds = self.unit_cell.dBds
        self.output.ddBdds = self.unit_cell.ddBdds
        self.to_hdf()

    def run_static(self):
        self.status.running = True
        self.ref_job_initialize()
        self.ref_job.set_fix_external(self.callback, overload_internal_fix_external=True)
        self.ref_job.run()
        self.status.collect = True
        self.collect_output()
        self.ref_job.interactive_close()
        self.run()

    def get_x_shift(self):
        x_v = -self.mass_ratios[-1] * self.total_displacements[-1]
        x_v -= self.input.unit_length * np.einsum(
            'i,ij->j',
            1 - self.mass_ratios[:-1],
            np.rint(self.total_displacements[:-1] / self.input.unit_length),
        )
        return self.structure.get_wrapped_coordinates(x_v)

    def append_displacement(self, x):
        if self.x_previous is None:
            self.x_previous = self.structure.positions
        if self.total_displacements is None:
            self.total_displacements = np.zeros_like(self.structure.positions)
        self.total_displacements += self.structure.get_wrapped_coordinates(x - self.x_previous)
        self.x_previous = x.copy()

    def callback(self, caller, ntimestep, nlocal, tag, x, fext):
        tags = tag.flatten().argsort()
        fext.fill(0)
        x_sorted = x[tags]
        x_shift = 0
        if self.input.track_vacancy and self.input.unit_length > 0:
            self.append_displacement(x_sorted)
            x_shift = self.get_x_shift()
        f = self.get_force(x_sorted[-1] - x_shift)
        fext[tags[-1]] += f
        fext[tags[:-1]] -= f / (len(tag) - 1)
        if ((ntimestep + 1) % self.input.update_every_n_steps) == 0:
            self.update_s(x_sorted[-1] - x_shift)

    def get_force(self, x):
        return self.unit_cell.get_force(x)

    def update_s(self, x):
        self.unit_cell.append_positions(x)

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
        if len(self.output.x) > 0:
            self.unit_cell._x_lst.extend(self.output.x)
            self.unit_cell.B = self.output.B
            self.unit_cell.dBds = self.output.dBds
            self.unit_cell.ddBdds = self.output.ddBdds

    def write_input(self):
        pass

    def get_energy(self, x, derivative=0):
        return self.unit_cell.get_energy(x, derivative=derivative)

    @property
    def _dE(self):
        return (self.input.sigma - self.input.min_sigma)**3 * (
            self.input.increment - self.input.min_increment
        )

    @property
    def _E_0(self):
        return self.input.min_sigma**3 * self.input.min_increment

    @property
    def _filling_coeff(self):
        V_tot = self.primitive_cell.get_volume()
        N_sym = len(self.unit_cell.symmetry.rotations)
        return (2 * np.pi)**1.5 / V_tot * N_sym

    def get_filling(self, n, diff=False):
        nn = np.array(n) / self.input.update_every_n_steps
        decay = self.input.sigma_decay**3 * self.input.increment_decay
        if diff:
            return self._filling_coeff * (self._dE * decay**nn + self._E_0)
        if decay < 1:
            return self._filling_coeff * (
                self._dE * (1 - decay**(nn + 1)) / (1 - decay) + nn * self._E_0
            )
        else:
            return self._filling_coeff * nn

    @property
    def max_force(self):
        return self.input.increment / self.input.sigma / np.exp(0.5)
