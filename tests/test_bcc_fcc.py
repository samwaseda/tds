# coding: utf-8
from tds.bcc_fcc import Diffusion
from pyiron_base._tests import TestWithCleanProject
import unittest
from sklearn.metrics import r2_score


class TestDiffusion(TestWithCleanProject):
    @classmethod
    def setUp(cls):
        cls.job = cls.project.create_job(Diffusion, 'job')
        cls.job.set_concentration(-0.5, temperature=1500)

    def test_dcdt(self):
        dmdx = self.job._get_gradient(self.job.chemical_potential) / self.job.kBT
        v_ref = self.job._get_gradient(self.job.get_D() * self.job.c * (1 - self.job.c) * dmdx)
        v_calc = self.job.dcdt
        self.assertGreater(r2_score(v_ref, v_calc), 0.999)

    def test_dDdx(self):
        v_calc = self.job.get_D(order=1)
        v_ref = self.job._get_gradient(self.job.get_D(order=0))
        self.assertGreater(r2_score(v_ref, v_calc), 0.999)

    def test_f(self):
        self.assertGreater(r2_score(
            self.job._get_f(order=1, absolute=False),
            self.job._get_gradient(self.job._get_f(absolute=False))
        ), 0.999)
        self.assertGreater(r2_score(
            self.job._get_f(order=2, absolute=False),
            self.job._get_laplace(self.job._get_f(absolute=False))
        ), 0.999)
        self.assertGreater(r2_score(
            self.job._get_f(order=3, absolute=False),
            self.job._get_laplace(self.job._get_f(order=1, absolute=False))
        ), 0.999)

    def test_dEdx(self):
        v_calc = self.job.get_E(order=1)
        v_ref = self.job._get_gradient(self.job.get_E(order=0))
        self.assertGreater(r2_score(v_ref, v_calc), 0.999)
        v_calc = self.job.get_E(order=2)
        v_ref = self.job._get_laplace(self.job.get_E(order=0))
        self.assertGreater(r2_score(v_ref, v_calc), 0.999)


if __name__ == "__main__":
    unittest.main()
