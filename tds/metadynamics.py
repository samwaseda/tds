from scipy.spatial import cKDTree
import numpy as np
from pyiron_base import PythonTemplateJob
from tds.grain_boundary import get_potential
from pyiron_atomistics.atomistics.structure.atoms import Atoms


class UnitCell:
    def __init__(self, unit_cell, sigma, increment, mesh_spacing=None, cutoff=None):
        self.unit_cell = unit_cell
        self.sigma = sigma
        self.increment = increment
        self.mesh_spacing = mesh_spacing
        if self.mesh_spacing is None:
            self.mesh_spacing = self.sigma / 4
        self._mesh = None
        self._tree = None
        self._symmetry = None
        self._x_repeat = None
        self.cutoff = cutoff
        self._x_lst = []
        self._cell_inv = None
        if self.cutoff is None:
            self.cutoff = 4 * sigma
        self.num_neighbors = int(1.1 * 4 / 3 * np.pi * self.cutoff**3 / self.mesh_spacing**3)
        self.dBds = np.zeros_like(self.mesh)
        self._symmetry = None

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
    def tree(self):
        if self._tree is None:
            self._tree = cKDTree(self.mesh.reshape(-1, 3))
        return self._tree

    @property
    def symmetry(self):
        if self._symmetry is None:
            self._symmetry = self.unit_cell.get_symmetry()
        return self._symmetry

    @property
    def x_repeat(self):
        if self._x_repeat is None:
            self._x_repeat = np.einsum(
                'j...,ji->...i',
                np.meshgrid(*3 * [[-1, 0, 1]]),
                self.unit_cell.cell
            ).reshape(-1, 3)
        return self._x_repeat

    def _get_symmetric_x(self, x_in):
        x = self.x_to_s(x_in)
        x = self.symmetry.generate_equivalent_points(x, return_unique=False)
        x = (x[:, np.newaxis, :] + self.x_repeat).reshape(-1, 3)
        return x[self.tree.query(x)[0] < self.cutoff]

    def append_positions(self, x_in):
        x = self._get_symmetric_x(x_in)
        self._x_lst.extend(x)
        dist, indices = self.tree.query(x, k=self.num_neighbors, distance_upper_bound=self.cutoff)
        cond = dist < np.inf
        dx = self.mesh.reshape(-1, 3)[indices[cond]] - x[np.indices(indices.shape)[0][cond]]
        unraveled_indices = np.unravel_index(indices[cond], self.mesh.shape[:-1])
        self.dBds[unraveled_indices] += self.increment / self.sigma**2 * dx * np.exp(
            -dist[cond]**2 / (2 * self.sigma**2)
        )[:, np.newaxis]

    def _get_index(self, x):
        return np.unravel_index(self.tree.query(self.x_to_s(x))[1], self.mesh.shape[:-1])

    def get_force(self, x):
        return self.dBds[self._get_index(x)]

    @property
    def x_lst(self):
        return np.array(self._x_lst)


class Metadynamics(PythonTemplateJob):
    def __init__(self, project, job_name):
        super().__init__(project, job_name)
        self.input.n_print = 1000
        self.input.number_of_steps = int(1e5)
        self.input.temperature = 300
        self.input.update_every_n_steps = 100
        self.input.increment = 0.001
        self.input.sigma = 0.38105
        self.input.cutoff = None
        self.input.mesh_spacing = None
        self.structure = None
        self._unit_cell = None

    @property
    def unit_cell(self):
        if self._unit_cell is None:
            gb = self.structure.get_symmetry().get_primitive_cell()
            self._unit_cell = UnitCell(
                unit_cell=gb,
                sigma=self.input.sigma,
                increment=self.input.increment,
                mesh_spacing=self.input.mesh_spacing,
                cutoff=self.input.cutoff
            )
        return self._unit_cell

    def run_static(self):
        x = np.random.permutation(self.structure.analyse.get_voronoi_vertices())[0]
        lmp = self.project.create.job.Lammps('lmp_{}'.format(self.job_name))
        lmp.structure = self.structure.copy()
        lmp.structure += lmp.structure[-1]
        lmp.structure[-1] = 'H'
        lmp.structure.positions[-1] = x
        lmp.potential = get_potential()
        lmp.server.run_mode.interactive = True
        lmp.calc_md(
            temperature=self.input.temperature,
            langevin=True,
            n_ionic_steps=1000,
            n_print=1000
        )
        lmp.run()
        lmp._generic_input["n_print"] = int(self.input.number_of_steps / 50)
        lmp._generic_input["n_ionic_steps"] = self.input.number_of_steps
        lmp._interactive_lib_command('fix 2 all external pf/callback 1 1')
        lmp._interactive_library.set_fix_external_callback("2", self.callback)
        lmp.run()
        lmp.interactive_close()
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
        with self.project_hdf5.open("input") as hdf5_input:
            self.structure.to_hdf(hdf5_input)

    def from_hdf(self, hdf=None, group_name=None):
        super().from_hdf(
            hdf=hdf,
            group_name=group_name
        )
        if (
            "structure" in self.project_hdf5["input"].list_groups()
        ):
            with self.project_hdf5.open("input") as hdf5_input:
                self.structure = Atoms().from_hdf(hdf5_input)
