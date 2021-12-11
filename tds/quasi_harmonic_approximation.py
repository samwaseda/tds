import numpy as np
import pint
from sklearn.linear_model import LinearRegression
from pyiron_atomistics.atomistics.master.parallel import AtomisticParallelMaster
from pyiron_base import JobGenerator


unit = pint.UnitRegistry()


def get_helmholtz_free_energy(frequencies, temperature, E_0=0):
    hn = (frequencies * 1e12 * unit.hertz * unit.planck_constant).to('eV').magnitude
    hn = hn[hn > 0]
    kBT = (temperature * unit.kelvin * unit.boltzmann_constant).to('eV').magnitude
    return 0.5 * np.sum(hn) + kBT * np.log(
        1 - np.exp(-np.einsum('i,...->...i', hn, 1 / kBT))
    ).sum(axis=-1) + E_0


def get_potential_energy(frequencies, temperature, E_0=0):
    hn = (frequencies * 1e12 * unit.hertz * unit.planck_constant).to('eV').magnitude
    kBT = (temperature * unit.kelvin * unit.boltzmann_constant).to('eV').magnitude
    return 0.5 * np.sum(hn) + np.sum(
        hn / (np.exp(-np.einsum('i,...->...i', hn, 1 / kBT)) - 1), axis=-1
    )


def generate_displacements(structure, symprec=1.0e-2):
    sym = structure.get_symmetry(symprec=symprec)
    indices, comp = np.unique(sym.arg_equivalent_vectors, return_index=True)
    ind_x, ind_y = np.unravel_index(comp, structure.positions.shape)
    displacements = np.zeros((len(ind_x),) + structure.positions.shape)
    displacements[np.arange(len(ind_x)), ind_x, ind_y] = 1
    return displacements


class Hessian:
    def __init__(self, structure, dx=0.01, symprec=1.0e-2, include_zero_strain=True):
        self.structure = structure.copy()
        self._symmetry = None
        self._permutations = None
        self.dx = dx
        self._displacements = []
        self._forces = None
        self._inequivalent_displacements = None
        self._symprec = symprec
        self._unit = None
        self._nu = None
        self._energy = None
        self.include_zero_strain = include_zero_strain

    @property
    def symmetry(self):
        if self._symmetry is None:
            self._symmetry = self.structure.get_symmetry(symprec=self._symprec)
        return self._symmetry

    @property
    def energy(self):
        return self._energy

    @energy.setter
    def energy(self, energy):
        if energy is not None and len(energy) != len(self.displacements):
            raise AssertionError('Energy shape does not match existing displacement shape')
        self._energy = np.array(energy)
        self._inequivalent_displacements = None
        self._nu = None

    @property
    def forces(self):
        return self._forces

    @forces.setter
    def forces(self, forces):
        if np.array(forces).shape != self.displacements.shape:
            raise AssertionError('Force shape does not match existing displacement shape')
        self._forces = np.array(forces)
        self._nu = None
        self._inequivalent_displacements = None

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
            self._inequivalent_displacements -= self.origin.flatten()
        return self._inequivalent_displacements

    @property
    def inequivalent_forces(self):
        forces = np.einsum(
            'nxy,lnmy->lnmx', self.symmetry.rotations, self.forces[:, self.symmetry.permutations]
        ).reshape(-1, np.prod(self.structure.positions.shape))
        return forces[self.inequivalent_indices]

    @property
    def _x_outer(self):
        return np.einsum('ik,ij->kj', *2 * [self.inequivalent_displacements])

    @property
    def hessian(self):
        H = -np.einsum(
            'kj,in,ik->nj',
            np.linalg.inv(self._x_outer),
            self.inequivalent_forces,
            self.inequivalent_displacements,
            optimize=True
        )
        return 0.5 * (H + H.T)

    @property
    def _mass_tensor(self):
        m = np.tile(self.structure.get_masses(), (3, 1)).T.flatten()
        return np.sqrt(m * m[:, np.newaxis])

    @property
    def vibrational_frequencies(self):
        if self._nu is None:
            H = self.hessian
            nu_square = (np.linalg.eigh(
                H / self._mass_tensor
            )[0] * unit.electron_volt / unit.angstrom**2 / unit.amu).to('THz**2').magnitude
            self._nu = np.sign(nu_square) * np.sqrt(np.absolute(nu_square)) / (2 * np.pi)
        return self._nu

    def get_free_energy(self, temperature):
        return get_helmholtz_free_energy(
            self.vibrational_frequencies[3:], temperature, self.min_energy
        )

    def get_potential_energy(self, temperature):
        return get_potential_energy(
            self.vibrational_frequencies[3:], temperature, self.min_energy
        )

    @property
    def minimum_displacements(self):
        return generate_displacements(structure=self.structure, symprec=self._symprec) * self.dx

    @property
    def displacements(self):
        if len(self._displacements) == 0:
            self.displacements = self.minimum_displacements
            if self.include_zero_strain:
                self.displacements = np.concatenate(
                    ([np.zeros_like(self.structure.positions)], self.displacements), axis=0
                )
        return np.asarray(self._displacements)

    @displacements.setter
    def displacements(self, d):
        self._displacements = np.asarray(d).tolist()
        self._inequivalent_displacements = None

    @property
    def _fit(self):
        E = self.energy + 0.5 * np.einsum('nij,nij->n', self.forces, self.displacements)
        E = np.repeat(E, len(self.symmetry.rotations))[self.inequivalent_indices]
        reg = LinearRegression()
        reg.fit(self.inequivalent_forces, E)
        return reg

    @property
    def min_energy(self):
        return self._fit.intercept_

    @property
    def origin(self):
        if self.energy is None or self.forces is None:
            return np.zeros_like(self.structure.positions)
        return 2 * self.symmetry.symmetrize_vectors(self._fit.coef_.reshape(-1, 3))

    @property
    def volume(self):
        return self.structure.get_volume()


class QHAJobGenerator(JobGenerator):
    @property
    def parameter_list(self):
        """

        Returns:
            (list)
        """
        cell, positions = self._master.get_supercells_with_displacements()
        return [[c, p] for c, p in zip(cell, positions)]

    def modify_job(self, job, parameter):
        job.structure.positions = parameter[0]
        job.structure.set_cell(parameter[1], scale_atoms=True)
        return job


class QuasiHarmonicApproximation(AtomisticParallelMaster):
    def __init__(self, project, job_name):
        super().__init__(project, job_name)
        self.__name__ = "QuasiHarmonicApproximation"
        self.__version__ = "0.0.1"
        self.input["displacement"] = (0.01, "Atom displacement magnitude")
        self.input['symprec'] = 1.0e-2
        self.input['include_zero_strain'] = True
        self.input["num_points"] = (11, "number of sample points")
        self.input["vol_range"] = (
            0.1,
            "relative volume variation around volume defined by ref_ham",
        )
        self._job_generator = QHAJobGenerator(self)
        self._hessian = None

    @property
    def strain_lst(self):
        if self.input['num_points'] == 1:
            return [0]
        return np.linspace(-1, 1, self.input['num_points']) * self.input['vol_range']

    @property
    def hessian(self):
        if self._hessian is None:
            self._hessian = Hessian(self.structure)
        return self._hessian

    def get_supercells_with_displacements(self):
        cell_lst, positions_lst = [], []
        for strain in self.strain_lst:
            for d in self.hessian.displacements:
                cell_lst.append(self.structure.cell * (1 + strain))
                positions_lst.append(self.structure.positions + d)
        return cell_lst, positions_lst

    def to_hdf(self, hdf=None, group_name=None):
        """
        Store the QHAJob in an HDF5 file

        Args:
            hdf (ProjectHDFio): HDF5 group object - optional
            group_name (str): HDF5 subgroup name - optional
        """
        super().to_hdf(hdf=hdf, group_name=group_name)
        if self.phonopy is not None and not self._disable_phonopy_pickle:
            with self.project_hdf5.open("output") as hdf5_output:
                hdf5_output["displacements"] = self.hessian.displacements

    def from_hdf(self, hdf=None, group_name=None):
        """
        Restore the PhonopyJob from an HDF5 file

        Args:
            hdf (ProjectHDFio): HDF5 group object - optional
            group_name (str): HDF5 subgroup name - optional
        """
        super().from_hdf(hdf=hdf, group_name=group_name)
        with self.project_hdf5.open("output") as hdf5_output:
            if 'displacements' in hdf5_output.list_nodes():
                self.hessian.displacements = hdf5_output['displacements']
