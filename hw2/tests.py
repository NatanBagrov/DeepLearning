from unittest import TestCase
from graph.Variable import Variable
from graph.Operation import Multiply
import numpy as np


class TestMultiply(TestCase):
    def test_forward(self):
        w = np.array([[1, 2, 3, 4],
                      [5, 6, 7, 8],
                      [9, 10, 11, 12]])
        x = np.array([[13, 14, ],
                      [15, 16, ],
                      [17, 18, ],
                      [19, 20, ]])

        wx_desired = np.array([[170, 180],
                               [426, 452],
                               [682, 724]])

        w_variable = Variable(w)
        x_variable = Variable(x)
        wx_variable = Multiply(w_variable, x_variable)

        wx_actual = wx_variable.forward()

        np.testing.assert_allclose(wx_actual, wx_desired)

    def test_backward_1(self):
        w = np.array([[1, 2], [3, 4], [5, 6], [7, 8]])
        x = np.array([[9, 10, 11], [12, 13, 14]])

        dl_dwx = np.ones((w.shape[0], x.shape[1]))

        w_variable = Variable(w)
        x_variable = Variable(x)
        wx_variable = Multiply(w_variable, x_variable)

        wx_desired = w @ x
        wx_actual = wx_variable.forward()

        np.testing.assert_allclose(wx_actual, wx_desired)
        wx_variable.backward(grad=dl_dwx)

        dl_dx_actual = x_variable.get_gradient()
        dl_dx_desired = np.array([[16, 16, 16], [20, 20, 20]])

        self.assertEqual(dl_dx_desired.shape, dl_dx_actual.shape)
        np.testing.assert_allclose(dl_dx_actual, dl_dx_desired)

    def test_backward_2(self):
        w = np.array([[1, 2], [3, 4], [5, 6], [7, 8]])
        x = np.array([[9, 10, 11], [12, 13, 14]])

        dl_dwx = np.arange(1, 1 + w.shape[0] * x.shape[1]).reshape(w.shape[0], x.shape[1])

        w_variable = Variable(w)
        x_variable = Variable(x)
        wx_variable = Multiply(w_variable, x_variable)

        wx_desired = w @ x
        wx_actual = wx_variable.forward()

        np.testing.assert_allclose(wx_actual, wx_desired)
        wx_variable.backward(grad=dl_dwx)

        dl_dw_actual = w_variable.get_gradient()
        dl_dw_desired = np.array([[1 * 9 + 2 * 10 + 3 * 11, 1 * 12 + 2 * 13 + 3 * 14],
                                  [4 * 9 + 5 * 10 + 6 * 11, 4 * 12 + 5 * 13 + 6 * 14],
                                  [7 * 9 + 8 * 10 + 9 * 11, 7 * 12 + 8 * 13 + 9 * 14],
                                  [10 * 9 + 11 * 10 + 12 * 11, 10 * 12 + 11 * 13 + 12 * 14]])

        self.assertEqual(dl_dw_desired.shape, dl_dw_actual.shape)
        np.testing.assert_allclose(dl_dw_actual, dl_dw_desired)
