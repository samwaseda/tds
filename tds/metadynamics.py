from scipy.spatial import cKDTree
import numpy as np
from pyiron_base.generic.datacontainer import DataContainer
from pyiron_atomistics.atomistics.job.interactivewrapper import InteractiveWrapper


class UnitCell:
    def __init__(self, unit_cell, sigma, increment, mesh_spacing=None, cutoff=None, symprec=1.0e-2):
        self.unit_cell = unit_cell
        self.sigma = sigma
        self.increment = increment
        self.mesh_spacing = mesh_spacing
        if self.mesh_spacing is None:
            self.mesh_spacing = self.sigma / 4
        self._mesh = None
        self._tree_mesh = None
        self._symmetry = None
        self._x_repeat = None
        self.cutoff = cutoff
        self._x_lst = []
        self._cell_inv = None
        if self.cutoff is None:
            self.cutoff = 4 * sigma
        self.num_neighbors = int(1.5 * 4 / 3 * np.pi * self.cutoff**3 / self.mesh_spacing**3)
        self.dBds = np.zeros_like(self.mesh)
        self._symmetry = None
        self._symprec = 1.0e-2

    def x_to_s(self, x):
        return self.unit_cell.get_wrapped_coordinates(x)

    @property
    def cell(self):
        return np.linalg.norm(self.unit_cell.cell, axis=-1)

    @property
    def mesh(self):
        if self._mesh is None:
            self._mesh = np.einsum('j...,ji->...i', np.meshgrid(
                *[np.linspace(0, 1, np.rint(c / self.mesh_spacing).astype(int)) for c in self.cell],
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

    def _get_symmetric_x(self, x_in):
        x = self.x_to_s(x_in)
        x = self.symmetry.generate_equivalent_points(x, return_unique=False)
        x = self.unit_cell.get_extended_positions(self.cutoff, positions=x)
        return x[self.tree_mesh.query(x)[0] < self.cutoff]

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
        self.dBds[unraveled_indices] += self.increment / self.sigma**2 * dx * np.exp(
            -dist**2 / (2 * self.sigma**2)
        )[:, np.newaxis]

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
        return int(1.5 * 4 / 3 * np.pi * self.cutoff**3 * rho)

    def get_energy(self, x):
        dist, indices = self.tree_output.query(
            self.x_to_s(x), k=self._num_neighbors_x_lst, distance_upper_bound=self.cutoff
        )
        return -self.increment * np.exp(-dist**2 / (2 * self.sigma**2)).sum(axis=-1)


class Metadynamics(InteractiveWrapper):
    def __init__(self, project, job_name):
        super().__init__(project, job_name)
        self.input = DataContainer(table_name='input')
        self.output = DataContainer(table_name='output')
        self.input.n_print = 1000
        self.input.number_of_steps = int(1e5)
        self.input.temperature = 300
        self.input.update_every_n_steps = 100
        self.input.increment = 0.001
        self.input.sigma = 0.38105
        self.input.cutoff = None
        self.input.mesh_spacing = None
        self.input.symprec = 1.0e-2
        self._unit_cell = None
        self.input.x_lst = []
        self.output.x_lst = []
        self._tree = None

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
                mesh_spacing=self.input.mesh_spacing,
                cutoff=self.input.cutoff,
                symprec=self.input.symprec
            )
            if len(self.input.x_lst) > 0:
                self.unit_cell.append_positions(self.input.x_lst, symmetrize=False)
        return self._unit_cell

    def run_static(self):
        self.ref_job.run()
        self.ref_job._generic_input["n_print"] = int(self.input.number_of_steps / 50)
        self.ref_job._generic_input["n_ionic_steps"] = self.input.number_of_steps
        self.ref_job._interactive_lib_command('fix 2 all external pf/callback 1 1')
        self.ref_job._interactive_library.set_fix_external_callback("2", self.callback)
        self.ref_job.run()
        self.ref_job.interactive_close()
        self.output.x_lst = self.unit_cell.x_lst
        self.status.finished = True
        self.to_hdf()

    def callback(self, caller, ntimestep, nlocal, tag, x, fext):
        tags = tag.flatten().argsort()
        fext.fill(0)
        f = self.get_force(x[tags[-1]])
        fext[tags[-1]] += f
        fext[tags[:-1]] -= f / (len(tag) - 1)
        if ((ntimestep + 1) % self.input.update_every_n_steps) == 0:
            self.update_s(x[tags[-1]])

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

    def write_input(self):
        pass

    def get_energy(self, x):
        return self.unit_cell.get_energy(x)
