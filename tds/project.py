import numpy as np
from pyiron_atomistics import Project as PyironProject
from tds.grain_boundary import GrainBoundary


class Project(PyironProject):
    def get_lattice_parameter(
        self, temperature=0, coeff=np.array([2.84356485e-08, 4.37466530e-05, 3.51975215e+00])
    ):
        return np.polyval(coeff, temperature)

    def get_structure(self, n_repeat=1, temperature=0, h_positions=None):
        structure = self.get_bulk(temperature=temperature).get_structure().repeat(n_repeat)
        if h_positions is not None:
            h_positions = np.atleast_2d(h_positions) * self.get_lattice_parameter(
                temperature=temperature
            )
            structure += self.create.structure.atoms(
                elements=len(h_positions) * ['H'],
                positions=h_positions,
                cell=structure.cell
            )
        return structure

    @staticmethod
    def get_potential():
        return '1995--Angelo-J-E--Ni-Al-H--LAMMPS--ipr1'

    def get_bulk(self, temperature=0):
        lmp = self.create.job.Lammps(('lmp', temperature))
        lmp.structure = self.create.structure.bulk(
            'Ni', cubic=True, a=self.get_lattice_parameter(temperature=temperature)
        )
        lmp.potential = self.get_potential()
        lmp.calc_minimize()
        if lmp.status.initialized:
            lmp.run()
        return lmp

    def get_energy(self, element, temperature=0, repeat=4):
        lmp_Ni = self.get_bulk(temperature=temperature)
        if element == 'Ni':
            return lmp_Ni['output/generic/energy_pot'][-1] / len(lmp_Ni.structure)
        elif element == 'H':
            lmp = self.create.job.Lammps('bulk_H')
            if lmp.status.initialized:
                lmp.potential = self.get_potential()
                lmp.structure = lmp_Ni.get_structure().repeat(repeat)
                a_0 = lmp_Ni.get_structure().cell[0, 0]
                lmp.structure += self.create.structure.atoms(positions=[[0, 0, 0.5 * a_0]], elements=['H'])
                lmp.calc_minimize()
                lmp.run()
            return lmp.output.energy_pot[-1] - len(lmp.structure.select_index('Ni')) * self.get_energy('Ni')
        else:
            raise ValueError('element not recognized')

    def get_grain_boundary(self, axis=[1, 0, 0], sigma=5, plane=[0, 1, 3], temperature=0, repeat=1):
        bulk = self.create.structure.bulk('Ni', cubic=True)
        self._grain_boundary = GrainBoundary(
            project=self,
            axis=axis,
            energy=self.get_energy('Ni'),
            sigma=sigma,
            plane=plane,
            repeat=repeat,
            bulk=bulk
        )
        structure = self._grain_boundary.grain_boundary.copy()
        structure.apply_strain(self.get_lattice_parameter(temperature) / bulk.cell[0, 0] - 1)
        return structure
