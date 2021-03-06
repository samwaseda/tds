# coding: utf-8
import numpy as np
import unittest
from pyiron_atomistics.atomistics.structure.factories.ase import AseFactory
from tds.metadynamics_3d import UnitCell
from scipy.stats import pearsonr


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
        dist, _ = self.unit_cell.tree_mesh.query(
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
        self.assertAlmostEqual(
            np.linalg.norm(x - self.unit_cell.x_lst, axis=-1).min(), 0
        )

    def test_get_neighbors(self):
        x = self.unit_cell._get_symmetric_x(np.random.random(3))
        dist, dx, indices = self.unit_cell._get_neighbors(x)
        self.assertTrue(np.allclose(dist, np.linalg.norm(dx, axis=-1)))
        x = self.unit_cell.x_to_s(np.random.random((1, 3)))
        dist, dx, indices = self.unit_cell._get_neighbors(x)
        self.assertTrue(
            np.allclose(
                dist,
                np.linalg.norm(self.unit_cell.mesh - x, axis=-1)[indices]
            )
        )
        self.assertTrue(np.allclose(dx, (self.unit_cell.mesh - x)[indices]))

    def test_get_energy(self):
        x = self.unit_cell.mesh[
            tuple(np.random.randint(self.unit_cell.mesh.shape[:-1]))
        ]
        self.unit_cell.append_positions(x)
        self.assertLessEqual(self.unit_cell.get_energy(x), -self.unit_cell.increment)
        x = self.unit_cell.x_to_s(x)
        dist, _ = self.unit_cell.tree_output.query(
            x,
            k=self.unit_cell._num_neighbors_x_lst,
            distance_upper_bound=self.unit_cell.cutoff
        )
        self.assertLess(
            np.sum(dist < np.inf, axis=-1).max(), self.unit_cell._num_neighbors_x_lst
        )

    def test_B(self):
        def get_diff(var, axis):
            return (np.roll(var, 1, axis=axis) - np.roll(var, -1, axis=axis)).flatten()
        x = self.unit_cell.mesh[
            tuple(np.random.randint(self.unit_cell.mesh.shape[:-1]))
        ]
        self.unit_cell.append_positions(x)
        ind = self.unit_cell._get_index(x)
        self.assertGreater(self.unit_cell.B[ind], self.unit_cell.increment)
        self.assertLess(np.linalg.eigh(self.unit_cell.ddBdds[ind])[0].max(), 0.)
        for i in range(3):
            dx = get_diff(self.unit_cell.mesh, i).reshape(-1, 3)
            indices = np.unique(
                np.round(np.linalg.norm(dx, axis=-1), decimals=8), return_inverse=True
            )[1]
            d_num = get_diff(self.unit_cell.B, i)[indices == 0]
            d_ana = np.sum(dx * self.unit_cell.dBds.reshape(-1, 3), axis=-1)[indices == 0]
            self.assertGreater(np.absolute(pearsonr(d_ana, d_num)[0]), 0.99)
            d_num = get_diff(self.unit_cell.dBds, i).reshape(-1, 3)[indices == 0].flatten()
            d_ana = np.einsum('...j,...ij->...i', dx, self.unit_cell.ddBdds.reshape(-1, 3, 3))
            self.assertGreater(
                np.absolute(pearsonr(d_ana[indices == 0].flatten(), d_num)[0]), 0.99
            )

    def test_get_force(self):
        x = np.random.random((1, 3))
        x = self.unit_cell.x_to_s(x)
        self.unit_cell.append_positions(x, symmetrize=False)
        force_max = self.unit_cell.increment / self.unit_cell.sigma * np.exp(-1 / 2)
        self.assertLess(
            abs(force_max - np.linalg.norm(self.unit_cell.dBds, axis=-1).max()),
            1.0e-4
        )

    def test_get_index(self):
        x = 100 * np.random.randn(3)
        ind_meta = self.unit_cell._get_index(x)
        x = self.unit_cell.x_to_s(x)
        ind_min = np.argmin(np.linalg.norm(self.unit_cell.mesh - x, axis=-1))
        ind_min = np.unravel_index(ind_min, self.unit_cell.mesh.shape[:-1])
        self.assertTrue(np.array_equal(ind_min, ind_meta))


if __name__ == "__main__":
    unittest.main()
