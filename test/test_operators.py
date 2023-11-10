import unittest
from void_migration import operators, params
import numpy as np


class TestOperators(unittest.TestCase):
    def test_simple_swap(self):
        src = [0, 0, 0]
        dst = [1, 1, 0]
        s = np.array([[[1], [2]], [[3], [4]]])
        arrays = [s, None]
        nu = np.zeros((2, 2))
        p = params.dict_to_class({"nm": 2})

        arrays, nu = operators.swap(src, dst, arrays, nu, p)

        self.assertEqual(arrays[0][0, 0, 0], 4)
        self.assertEqual(arrays[1], None)
        self.assertTrue(np.allclose(nu, np.array([[0.5, 0.0], [0.0, -0.5]])))


if __name__ == "__main__":
    unittest.main()
