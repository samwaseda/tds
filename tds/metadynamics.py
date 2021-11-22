from scipy.spatial import cKDTree
import numpy as np


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
        if self.cutoff is None:
            self.cutoff = 4 * sigma
        self.num_neighbors = int(1.1 * 4 / 3 * np.pi * self.cutoff**3 / self.mesh_spacing**3)
        self.dBds = np.zeros_like(self.mesh)

    def x_to_s(self, x):
        return x - np.floor(x / self.unit_cell.cell.diagonal()) * self.unit_cell.cell.diagonal()

    @property
    def cell(self):
        return self.unit_cell.cell.diagonal()

    @property
    def mesh(self):
        if self._mesh is None:
            self._mesh = np.stack(np.meshgrid(
                *[np.linspace(0, c, np.rint(c / self.mesh_spacing).astype(int)) for c in self.cell],
                indexing='ij'
            ), axis=-1)
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
            self._x_repeat = np.stack(
                np.meshgrid(*3 * [[-1, 0, 1]]), axis=-1
            ).reshape(-1, 3) * self.cell
        return self._x_repeat

    def _get_symmetric_x(self, x_in):
        x = self.x_to_s(x_in)
        x = self.symmetry.generate_equivalent_points(x, return_unique=False)
        x = (x[:, np.newaxis, :] + self.x_repeat).reshape(-1, 3)
        return x[np.logical_and(
            np.min(x, axis=-1) > -self.cutoff, np.max(x - self.cell, axis=-1) < self.cutoff
        )]

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
