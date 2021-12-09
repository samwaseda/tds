import numpy as np
import pint


def generate_displacements(structure, symprec=1.0e-2):
    sym = structure.get_symmetry(symprec=symprec)
    indices, comp = np.unique(sym.arg_equivalent_vectors, return_index=True)
    ind_x, ind_y = np.unravel_index(comp, structure.positions.shape)
    displacements = np.zeros((len(ind_x),) + structure.positions.shape)
    displacements[np.arange(len(ind_x)), ind_x, ind_y] = 1
    return displacements


class Hessian:
    def __init__(self, structure, dx=0.01, symprec=1.0e-2):
        self.structure = structure.copy()
        self._symmetry = None
        self._permutations = None
        self.dx = dx
        self._displacements = []
        self._forces = None
        self._inequivalent_displacements = None
        self._symprec = symprec
        self._unit = None

    @property
    def symmetry(self):
        if self._symmetry is None:
            self._symmetry = self.structure.get_symmetry(symprec=self._symprec)
        return self._symmetry

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
        return np.einsum(
            'nxy,lnmy->lnmx', self.symmetry.rotations,
            self.displacements[:, self.symmetry.permutations]
        ).reshape(-1, np.prod(self.structure.positions.shape))

    @property
    def inequivalent_indices(self):
        return np.unique(self.all_displacements, axis=0, return_index=True)[1]

    @property
    def inequivalent_displacements(self):
        if self._inequivalent_displacements is None:
            self._inequivalent_displacements = self.all_displacements[self.inequivalent_indices]
        return self._inequivalent_displacements

    @property
    def inequivalent_forces(self):
        forces = np.einsum(
            'nxy,lnmy->lnmx', self.symmetry.rotations, self.forces[:, self.symmetry.permutations]
        ).reshape(-1, np.prod(self.structure.positions.shape))
        return forces[self.inequivalent_indices]

    @property
    def _x_outer(self):
        return np.einsum(
            'ik,ij->kj',
            self.inequivalent_displacements,
            self.inequivalent_displacements, optimize=True
        )

    def get_hessian(self, forces=None):
        if forces is None and self.forces is None:
            raise AssertionError('Forces not set yet')
        if forces is not None:
            self.forces = forces
        H = -np.einsum(
            'kj,in,ik->nj',
            np.linalg.inv(self._x_outer),
            self.inequivalent_forces,
            self.inequivalent_displacements,
            optimize=True
        )
        return 0.5 * (H + H.T)

    @property
    def _to_THz(self):
        return np.sqrt((
            1 * (self.unit.electron_volt / self.unit.angstrom**2 / self.unit.amu)
        ).to('THz**2').magnitude)

    @property
    def _mass_tensor(self):
        m = np.tile(self.structure.get_masses(), (3, 1)).T.flatten()
        return np.sqrt(m * m[:, np.newaxis])

    def get_vibrational_frequencies(self, forces=None):
        H = self.get_hessian(forces=forces)
        nu_square = np.linalg.eigh(H / self._mass_tensor)[0]
        return np.sign(nu_square) * np.sqrt(np.absolute(nu_square)) / (2 * np.pi) * self._to_THz

    @property
    def unit(self):
        if self._unit is None:
            self._unit = pint.UnitRegistry()
        return self._unit

    def get_free_energy(self, temperature, forces=None):
        nu = self.get_vibrational_frequencies(forces=forces)
        hn = (nu[3:] * 1e12 * self.unit.hertz * self.unit.planck_constant).to('eV').magnitude
        kBT = (temperature * self.unit.kelvin * self.unit.boltzmann_constant).to('eV').magnitude
        return 0.5 * np.sum(hn) + kBT * np.log(
            1 - np.exp(-np.einsum('i,...->i...', hn, 1 / kBT))
        ).sum(axis=0)

    @property
    def minimum_displacements(self):
        return generate_displacements(structure=self.structure, symprec=self._symprec) * self.dx

    @property
    def displacements(self):
        if len(self._displacements) == 0:
            self.displacements = self.minimum_displacements
        return np.asarray(self._displacements)

    @displacements.setter
    def displacements(self, d):
        self._displacements = np.asarray(d).tolist()
        self._inequivalent_displacements = None
