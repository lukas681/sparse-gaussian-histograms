import unittest
import main.hist.gshm as gshm
import main.hist.cgshm as cgshm

class MyTestCase(unittest.TestCase):
    def test_compute_add_deltas_tau(self):
        print(gshm.compute_tau_add_deltas(0.00001, 0.349, 51915, 2300))
        self.assertEqual(True, True)  # add assertion here
    def testr_compute_cgshm_tighter(self):
        print(gshm)
        self.assertEqual(True, True)

if __name__ == '__main__':
    unittest.main()
