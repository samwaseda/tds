from pyiron_atomistics.atomistics.structure.atoms import Atoms
import numpy as np


class GrainBoundary:
    def __init__(
        self, project, bulk, energy, axis=[1, 0, 0], sigma=5, plane=[0, 1, 3], repeat=1
    ):
        self.energy = energy
        self._energy_lst = []
        self._structure_lst = []
        self.project = project
        self.axis = axis
        self.sigma = sigma
        self.plane = plane
        self.repeat = repeat
        self.bulk = bulk

    @property
    def gb_energy(self):
        if len(self._energy_lst) == 0:
            self.run()
        return self._energy_lst.min()

    @property
    def grain_boundary(self):
        if len(self._structure_lst) == 0:
            self.run()
        return self._structure_lst[self._energy_lst.argmin()]

    def run(self):
        for i in range(2):
            for j in range(2):
                gb = self.project.create.structure.aimsgb.build(
                    self.axis, self.sigma, self.plane, self.bulk, delete_layer='{0}b{1}t{0}b{1}t'.format(i, j),
                    uc_a=self.repeat, uc_b=self.repeat
                )
                lmp = self.project.create.job.Lammps(
                    ('lmp', self.repeat, *self.axis, self.sigma, *self.plane, i, j)
                )
                if lmp.status.initialized:
                    lmp.structure = gb
                    lmp.potential = self.project.get_potential()
                    lmp.calc_minimize(pressure=0)
                    lmp.run()
                E = lmp.output.energy_pot[-1] - self.energy * len(gb)
                cell = lmp.output.cells[-1].diagonal()
                self._energy_lst.append(E / cell.prod() * np.max(cell) / 2)
                self._structure_lst.append(lmp.get_structure())
        self._energy_lst = np.asarray(self._energy_lst)


class Interstitials:
    def __init__(self, ref_structure, positions, energy=None, eps=1, cutoff_radius=1.6421):
        self.ref_structure = ref_structure
        self.structure = ref_structure.copy()
        self.labels = None
        self._energy = None
        self.eps = eps
        self.positions = positions
        self.bulk_layer = 3
        # 3/4 tetra-octa + 1/4 octa-octa = 3 A
        self.cutoff_radius = cutoff_radius
        if energy is not None:
            self.energy = energy
        self._neigh = None
        self._pairs = None

    def set_repeat(self, repeat):
        self.ref_structure = self.structure.repeat(repeat)
        self.structure = self.structure.repeat(repeat)
        self.positions = self.structure.positions
        self.energy = np.append(*np.prod(repeat) * [self.energy])

    @property
    def energy(self):
        return self._energy

    @energy.setter
    def energy(self, new_energy):
        self._energy = np.array(new_energy)[np.argsort(self.labels)]

    @property
    def positions(self):
        return self.structure.positions

    @positions.setter
    def positions(self, new_positions):
        positions, self.labels = self.structure.analyse.cluster_positions(
            new_positions, return_labels=True, eps=self.eps
        )
        self.structure = Atoms(
            elements=len(positions) * ['H'], cell=self.structure.cell, positions=positions, pbc=True
        )

    def append_positions(self, positions, energy=None):
        positions = np.append(self.positions, np.atleast_2d(positions), axis=0)
        self.positions = positions
        if energy is not None:
            self.energy = np.append(self.energy, energy)

    @property
    def equivalent_atoms(self):
        return self.structure.get_symmetry().arg_equivalent_atoms

    @property
    def defect_counter(self):
        neigh = self.ref_structure.get_neighborhood(self.positions, num_neighbors=None, cutoff_radius=self.bulk_layer)
        counter = np.zeros(len(self.positions))
        np.add.at(
            counter, neigh.flattened.atom_numbers,
            self.ref_structure.analyse.pyscal_cna_adaptive(mode='str')[neigh.flattened.indices] == 'others'
        )
        return counter

    @property
    def neigh(self):
        if self._neigh is None:
            self._neigh = self.structure.get_neighbors(
                num_neighbors=None, cutoff_radius=self.cutoff_radius
            )
        return self._neigh

    @property
    def pairs(self):
        if self._pairs is None:
            self._pairs = np.stack((
                self.neigh.flattened.atom_numbers, self.neigh.flattened.indices
            ), axis=-1)
            self._pairs = self._pairs[np.diff(self._pairs, axis=-1).squeeze() > 0]
        return self._pairs

    @property
    def minimum_pairs(self):
        return self.pairs[
            np.unique(self.equivalent_atoms[self.pairs], axis=0, return_index=True)[1]
        ]
