# coding: utf-8
from scipy.stats import pearsonr
import numpy as np
import unittest
from tds.metadynamics_1d import Metadynamics
from tds.project import Project


class TestUnitCell(unittest.TestCase):
    @classmethod
    def setUp(cls):
        cls.project = Project('DIFFUSION')
        cls.lammps = cls.project.create.job.Lammps('lmp')
        axis = 2
        cls.lammps.structure = cls.project.create.structure.bulk('Ni', cubic=True).repeat(5)
        del cls.lammps.structure[cls.lammps.structure.analyse.get_layers()[:, axis] == 0]
        cls.lammps.potential = cls.project.get_potential()
        cls.lammps.interactive_open()
        cls.meta = cls.lammps.create_job(Metadynamics, 'meta')
        cls.meta.input.axis = axis
        cls.meta._initialize_potentials()

    @classmethod
    def tearDown(cls):
        cls.project.remove(enable=True)

    def test_B(self):
        x = np.random.random() * self.lammps.structure.cell[tuple(2 * [self.meta.input.axis])]
        self.meta.update_s(x)
        self.assertGreater(
            pearsonr(np.gradient(self.meta.output.dBds, self.meta.mesh), self.meta.output.ddBdds)[0],
            0.999
        )
        self.assertGreater(
            pearsonr(np.gradient(self.meta.output.B, self.meta.mesh), self.meta.output.dBds)[0],
            0.999
        )

    def test_B(self):
        x = np.random.random() * self.lammps.structure.cell[tuple(2 * [self.meta.input.axis])]
        dx = x * 0.1
        x *= 0.9
        self.meta.update_s(x)
        self.assertGreater(self.meta.get_force(x + dx), 0)
        self.assertLess(self.meta.get_force(x - dx), 0)


if __name__ == "__main__":
    unittest.main()
