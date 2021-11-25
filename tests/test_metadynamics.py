# coding: utf-8

import numpy as np
import unittest
from pyiron_atomistics.atomistics.structure.factories.ase import AseFactory
from tds.metadynamics import UnitCell


class TestUnitCell(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.primitive_cell = AseFactory().bulk('Ni')

    @classmethod
    def setUp(cls):
        cls.unit_cell = UnitCell(cls.primitive_cell, 0.1, 0.001)

    def test_x_to_s(self):
        x = self.primitive_cell.cell @ np.random.random(3)
        randint = np.random.randint(10, size=3) - 5
        x_repeat = np.sum(randint[:, None] * self.primitive_cell.cell, axis=0)
        self.assertTrue(
            np.allclose(self.unit_cell.x_to_s(x), self.unit_cell.x_to_s(x + x_repeat))
        )

    def test_num_neighbors(self):
        x_center = np.sum(0.5 * self.primitive_cell.cell, axis=-1)
        x = self.unit_cell.x_to_s(np.random.randn(10, 3) + x_center)
        dist, _ = self.unit_cell.tree.query(
            x, k=self.unit_cell.num_neighbors, distance_upper_bound=self.unit_cell.cutoff
        )
        self.assertLess(np.sum(dist < np.inf, axis=-1).max(), self.unit_cell.num_neighbors)

    def test_get_symmetric_x(self):
        x = self.unit_cell._get_symmetric_x(np.random.random(3))
        self.assertEqual(
            len(np.unique(self.unit_cell.symmetry.get_arg_equivalent_sites(x))), 1
        )

    def test_append_position(self):
        x = np.random.random(3)
        self.unit_cell.append_positions(x)
        self.assertAlmostEqual(np.linalg.norm(x-self.unit_cell.x_lst, axis=-1).min(), 0)


if __name__ == "__main__":
    unittest.main()
