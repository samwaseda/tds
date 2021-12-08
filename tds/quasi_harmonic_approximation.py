import numpy as np
from scipy.spatial import cKDTree
import pint


class Hessian:
    def __init__(self, structure, dx=0.01):
        self.structure = structure.copy()
        self._symmetry = None
        self._permutations = None
        self.dx = dx
        self._displacements = []
        self._all_displacements = None
        self._forces = None
        self._unique_atoms = None

    @property
    def unique_atoms(self):
        if self._unique_atoms is None:
            self._unique_atoms = np.unique(self.symmetry.arg_equivalent_atoms)
        return self._unique_atoms

    @property
    def symmetry(self):
        if self._symmetry is None:
            self._symmetry = self.structure.get_symmetry()
        return self._symmetry

    @property
    def permutations(self):
        if self._permutations is None:
            epsilon = 1.0e-8
            x_scale = self.structure.get_scaled_positions()
            x = np.einsum(
                'nxy,my->mnx', self.symmetry.rotations, x_scale
            ) + self.symmetry.translations
            if any(self.structure.pbc):
                x[:, :, self.structure.pbc] -= np.floor(x[:, :, self.structure.pbc] + epsilon)
            x = np.einsum('nmx->mnx', x)
            tree = cKDTree(x_scale)
            self._permutations = np.argsort(tree.query(x)[1], axis=-1)
        return self._permutations

    def _get_equivalent_vector(self, v, indices=None):
        result = np.einsum('nxy,nmy->nmx', self.symmetry.rotations, v[self.permutations])
        if indices is None:
            indices = np.sort(np.unique(result, return_index=True, axis=0)[1])
        return result[indices], indices

    @property
    def forces(self):
        return self._forces

    @forces.setter
    def forces(self, forces):
        if np.array(forces).shape != self.displacements.shape:
            raise AssertionError('Force shape does not match existing displacement shape')
        self._forces = np.array(forces)

    @property
    def all_displacements(self):
        if self._all_displacements is None:
            self._all_displacements = np.einsum(
                'nxy,lnmy->lnmx', self.symmetry.rotations, self.displacements[:, self.permutations]
            ).reshape(-1, np.prod(self.structure.positions.shape))
        return self._all_displacements

    @property
    def inequivalent_indices(self):
        return np.unique(self.all_displacements, axis=0, return_index=True)[1]

    @property
    def inequivalent_displacements(self):
        return self.all_displacements[self.inequivalent_indices]

    @property
    def inequivalent_forces(self):
        forces = np.einsum(
            'nxy,lnmy->lnmx', self.symmetry.rotations, self.forces[:, self.permutations]
        ).reshape(-1, np.prod(self.structure.positions.shape))
        return forces[self.inequivalent_indices]

    @property
    def _sum_displacements(self):
        if len(self._displacements) == 0:
            displacements = np.zeros((len(self.unique_atoms),) + self.structure.positions.shape)
            displacements[np.arange(len(self.unique_atoms)), self.unique_atoms, 0] = self.dx
            self.displacements = displacements
        return np.absolute(self.inequivalent_displacements).sum(axis=0).reshape(
            self.structure.positions.shape
        )

    @property
    def _next_displacement(self):
        ix = np.stack(np.where(self._sum_displacements == 0), axis=-1)
        if len(ix) == 0:
            return None
        ix = ix[np.unique(ix[:, 0], return_index=True)[1]]
        ix = ix[np.any(ix[:, 0, None] == self.unique_atoms, axis=1)]
        displacements = np.zeros((len(ix), ) + self.structure.positions.shape)
        displacements[np.arange(len(ix)), ix[:, 0], ix[:, 1]] = self.dx
        return displacements.tolist()

    def _generate_displacements(self):
        for _ in range(np.prod(self.structure.positions.shape)):
            displacement = self._next_displacement
            if displacement is None:
                break
            self._displacements.extend(displacement)
            self._all_displacements = None

    def get_hessian(self, forces=None):
        if forces is None and self.forces is None:
            raise AssertionError('Forces not set yet')
        if forces is not None:
            self.forces = forces
        X = np.einsum(
            'ik,ij->kj',
            self.inequivalent_displacements,
            self.inequivalent_displacements, optimize=True
        )
        H = -np.einsum(
            'kj,in,ik->nj',
            np.linalg.inv(X),
            self.inequivalent_forces,
            self.inequivalent_displacements,
            optimize=True
        )
        return 0.5 * (H + H.T)

    @property
    def _to_THz(self):
        u = pint.UnitRegistry()
        return np.sqrt((1 * (u.electron_volt / u.angstrom**2 / u.amu)).to('THz**2').magnitude)

    @property
    def _mass_tensor(self):
        m = np.tile(self.structure.get_masses(), (3, 1)).T.flatten()
        return np.sqrt(m * m[:, np.newaxis])

    def get_vibrational_frequencies(self, forces=None):
        H = self.get_hessian(forces=forces)
        nu_square = np.linalg.eigh(H / self._mass_tensor)[0]
        return np.sign(nu_square) * np.sqrt(np.absolute(nu_square)) / (2 * np.pi) * self._to_THz

    @property
    def displacements(self):
        if len(self._displacements) == 0:
            self._generate_displacements()
        return np.asarray(self._displacements)

    @displacements.setter
    def displacements(self, d):
        self._displacements = np.asarray(d).tolist()
        self._all_displacements = None
