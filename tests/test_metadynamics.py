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
        x = self.primitive_cell.cell@np.random.random(3)
        randint = np.random.randint(10, size=3) - 5
        x_repeat = np.sum(randint[:, None] * self.primitive_cell.cell, axis=0)
        self.assertTrue(np.allclose(self.unit_cell.x_to_s(x), self.unit_cell.x_to_s(x + x_repeat)))


if __name__ == "__main__":
    unittest.main()
