import unittest
from math import sqrt
import main.hist.gshm as gshm
import main.hist.cgshm as cgshm

class MyTestCase(unittest.TestCase):
    def test_compute_add_deltas_tau(self):
        # print(gshm.compute_tau_add_deltas(0.00001, 0.349, 51915, 2300))
        self.assertEqual(True, True)  # add assertion here
    def test_compute_cgshm_tighter(self):
        print(gshm)
        self.assertEqual(True, True)
    def test_minimum_amount_pof_noise(self):
        k = 51914
        epsilon = 0.347
        delta = 10**-5
        mu = cgshm.minimum_amount_of_noise(sqrt(epsilon), epsilon, delta)
        min_sigma_reference = sqrt(k)/mu # Doing transformation after is numerically more stable.
        min_sigma_we = sqrt(k + sqrt(k)) / (2*mu) # Doing transformation after is numerically more stable.
        print(min_sigma_reference)
        self.assertTrue(min_sigma_reference >2240 and min_sigma_reference <2246) # Old threhold ok?
        print(f'we: {min_sigma_we}')
        self.assertTrue(min_sigma_we>1100 and min_sigma_we <1200) # CHeck this later.

    def test_check_gshm_validity(self):
        k, epsilon, delta = 51914, 0.349,10**-5
        mu = cgshm.minimum_amount_of_noise(sqrt(epsilon), epsilon, delta) ## Pseudo function
        sig = sqrt(k)/mu # Doing transformation after is numerically more stable.
        tau = 16176 ## Should be fine.
        self.assertTrue(gshm.check_validity(k, sig, tau, epsilon, delta ))

    def test_find_tau_gshm(self):
        k, epsilon, delta = 51914, 0.349,10**-5
        res = gshm.compute_threshold_curve(delta, epsilon, k)
        print(f'{res[1]}, {res[0]}')
        self.assertTrue(True, True)



if __name__ == '__main__':
    unittest.main()
