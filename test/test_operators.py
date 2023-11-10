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

    def test_simple_solid_fraction(self):
        s = np.array([[[2,np.nan],[3,1]], [[1,1],[4,np.nan]]])

        s_bar = operators.get_solid_fraction(s,1,1)
        
        self.assertEqual(s_bar,0.5)

    def test_simple_get_average(self):
        s = np.array([[[2,np.nan,np.nan],[3,3,3]], [[1,1,np.nan],[4,np.nan,np.nan]]])

        s_mean = operators.get_average(s)

        self.assertEqual(s_mean[0],1.5)
        self.assertEqual(s_mean[1],3.5)

    def test_get_hyperbolic_average(self):
        s = np.array([[[2,np.nan],[3,1]], [[1,1],[4,np.nan]]])

        s_inv_bar = operators.get_hyperbolic_average(s)
        self.assertEqual(s_inv_bar[0][0],2)
        self.assertEqual(s_inv_bar[1][1],4)


if __name__ == "__main__":
    unittest.main()
