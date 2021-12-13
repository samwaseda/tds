from scipy.spatial import cKDTree
import numpy as np
from pyiron_base.generic.datacontainer import DataContainer
from pyiron_atomistics.atomistics.job.interactivewrapper import InteractiveWrapper
from tds.gaussian_process import GaussianProcess


class UnitCell:
    def __init__(self, unit_cell, sigma, increment, spacing=None, cutoff=None, symprec=1.0e-2):
        self.unit_cell = unit_cell
        self.sigma = sigma
        self.increment = increment
        self.spacing = spacing
        if self.spacing is None:
            self.spacing = self.sigma / 4
        self._mesh = None
        self._tree_mesh = None
        self._symmetry = None
        self._x_repeat = None
        self.cutoff = cutoff
        self._x_lst = []
        self._cell_inv = None
        if self.cutoff is None:
            self.cutoff = 4 * sigma
        self.num_neighbors = int(1.5 * 4 / 3 * np.pi * self.cutoff**3 / self.spacing**3)
        self.dBds = np.zeros_like(self.mesh)
        self._symmetry = None
        self._symprec = 1.0e-2
        self._gaussian_process = None

    def x_to_s(self, x):
        return self.unit_cell.get_wrapped_coordinates(x)

    @property
    def cell(self):
        return np.linalg.norm(self.unit_cell.cell, axis=-1)

    @property
    def mesh(self):
        if self._mesh is None:
            self._mesh = np.einsum('j...,ji->...i', np.meshgrid(
                *[np.linspace(0, 1, np.rint(c / self.spacing).astype(int)) for c in self.cell],
                indexing='ij'
            ), self.unit_cell.cell)
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
        self._x_lst.extend(x)
        dist, dx, unraveled_indices = self._get_neighbors(x)
        np.add.at(
            self.dBds,
            unraveled_indices,
            self.increment / self.sigma**2 * dx * np.exp(
                -dist**2 / (2 * self.sigma**2)
            )[:, np.newaxis]
        )

    def _get_index(self, x):
        return np.unravel_index(self.tree_mesh.query(self.x_to_s(x))[1], self.mesh.shape[:-1])

    def get_force(self, x):
        return self.dBds[self._get_index(x)]

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

    @property
    def gaussian_process(self):
        if self._gaussian_process is None:
            self._gaussian_process = GaussianProcess(self._get_energy, max_error=1.0e-2)
        return self._gaussian_process

    def get_energy(self, x, reset_gp=False):
        if reset_gp:
            self.gaussian_process = None
        s_in = self.x_to_s(x)
        s = np.asarray(s_in).reshape(-1, np.shape(s_in)[-1])
        while True:
            ss = self.gaussian_process.get_arg_max_error(s)
            if ss is None:
                break
            self.gaussian_process.append(ss)
            self.gaussian_process.replicate(self._get_symmetric_x(ss, cutoff=self.cutoff/4))
        return self.gaussian_process.predict(s).reshape(s_in.shape[:-1])

    def _get_energy(self, x):
        dist, indices = self.tree_output.query(
            x, k=self._num_neighbors_x_lst, distance_upper_bound=self.cutoff
        )
        return -self.increment * np.exp(-dist**2 / (2 * self.sigma**2)).sum(axis=-1)

    def get_gradient(self, x):
        return -self.gaussian_process.get_gradient(x)

    def get_hessian(self, x):
        return -self.gaussian_process.get_hessian(x)


class Metadynamics(InteractiveWrapper):
    def __init__(self, project, job_name):
        super().__init__(project, job_name)
        self.input = DataContainer(table_name='input')
        self.output = DataContainer(table_name='output')
        self.input.n_print = 1000
        self.input.number_of_steps = int(1e5)
        self.input.temperature = 300
        self.input.update_every_n_steps = 100
        self.input.increment = 0.0001
        self.input.sigma = 0.38105
        self.input.cutoff = None
        self.input.spacing = None
        self.input.symprec = 1.0e-2
        self.input.e_prefactor = 3.0227679
        self.input.e_decay = 1.30318
        self.input.num_neighbors = 20
        self.input.unit_length = 0
        self.input.track_vacancy = False
        self._unit_cell = None
        self.input.x_lst = []
        self.output.x_lst = []
        self._tree = None
        self._mass_ratios = None
        self.total_displacements = None
        self.x_previous = None

    def prefill_histo(self):
        if not np.isclose(self.input.e_prefactor, 0):
            neigh = self.unit_cell.unit_cell.get_neighborhood(
                self.unit_cell.mesh, num_neighbors=self.input.num_neighbors
            )
            self.unit_cell.dBds = 2 * self.input.e_prefactor * self.input.e_decay * np.einsum(
                '...ni,...n->...i',
                neigh.vecs,
                np.exp(-self.input.e_decay * neigh.distances**2)
            )

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
                symprec=self.input.symprec
            )
            self.prefill_histo()
            if len(self.input.x_lst) > 0:
                self.unit_cell.append_positions(self.input.x_lst, symmetrize=False)
        return self._unit_cell

    @property
    def mass_ratios(self):
        if self._mass_ratios is None:
            self._mass_ratios = self.structure.get_masses()
            self._mass_ratios /= self._mass_ratios.sum()
        return self._mass_ratios

    def run_static(self):
        self.status.running = True
        self.ref_job_initialize()
        self.ref_job.set_callback(self.callback, overload_internal_callback=True)
        self.ref_job.run()
        self.status.collect = True
        self.ref_job.interactive_close()
        self.output.x_lst = self.unit_cell.x_lst
        self.status.finished = True
        self.to_hdf()

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
        if len(self.output.x_lst) > 0:
            self.unit_cell._x_lst.extend(self.output.x_lst)

    def write_input(self):
        pass

    def _get_prefill_energy(self, x):
        d = self.unit_cell.unit_cell.get_neighborhood(
            x, num_neighbors=self.input.num_neighbors
        ).distances
        return self.input.e_prefactor * np.exp(-self.input.e_decay * d**2).sum(axis=-1)

    def _get_prefill_gradient(self, x):
        neigh = self.unit_cell.unit_cell.get_neighborhood(
            x, num_neighbors=self.input.num_neighbors
        )
        return 2 * self.input.e_decay * self.input.e_prefactor * np.einsum(
            '...ij,...i->...j', neigh.vecs, np.exp(-self.input.e_decay * neigh.distances**2)
        )

    def _get_prefill_hessian(self, x):
        neigh = self.unit_cell.unit_cell.get_neighborhood(
            x, num_neighbors=self.input.num_neighbors
        )
        H = 4 * self.input.e_decay**2 * np.einsum(
            '...i,...j->...ij', neigh.vecs, neigh.vecs
        ) + 2 * self.input.e_decay**2
        return self.input.e_prefactor * np.einsum(
            '...ijk,...i->...jk', H, np.exp(-self.input.e_decay * neigh.distances**2)
        )

    def get_energy(self, x):
        return self.unit_cell.get_energy(x) + self._get_prefill_energy(x)

    def get_gradient(self, x):
        return self.unit_cell.get_gradient(x) + self._get_prefill_gradient(x)

    def get_hessian(self, x):
        return self.unit_cell.get_hessian(x) + self._get_prefill_hessian(x)

    @property
    def filling_rate(self):
        dEV = (np.sqrt(np.pi) * self.input.sigma)**3 * self.input.increment
        V_tot = self.primitive_cell.get_volume()
        N_sym = len(self.unit_cell.symmetry.rotations)
        N_freq = self.input.update_every_n_steps
        return dEV / V_tot * N_sym / N_freq
