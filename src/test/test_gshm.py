import unittest
from src.main.GSHM import compute_tau_add_deltas


class MyTestCase(unittest.TestCase):
    def test_compute_add_deltas_tau(self):
        print(compute_tau_add_deltas(0.00001,0.349, 51915, 2300))
        self.assertEqual(True, True)  # add assertion here


if __name__ == '__main__':
    unittest.main()
