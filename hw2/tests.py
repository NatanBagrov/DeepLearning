from random import random
from unittest import TestCase

from graph.UnaryOperations import ReduceMean, ReduceSum, ReduceSize, Transpose
from graph.Variable import Variable
from graph.BinaryOperations import Add, Multiply, HadamardMult
from utils.LossFunctions import MSE
import numpy as np
from sklearn.metrics import mean_squared_error

from utils.RegularizationMethods import L1, L2


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

        # self.fail()


class TestAdd(TestCase):
    def test_forward(self):
        x = np.array([[1, 2], [3, 4], [5, 6]])
        b = np.array([[7, 8], [9, 10], [11, 12]])

        dl_dxb = np.array([[13, 14], [15, 16], [17, 18]])

        x_variable = Variable(x)
        b_variable = Variable(b)
        wx_variable = Add(x_variable, b_variable)

        wx_variable.forward()
        wx_variable.backward(dl_dxb)

        dl_dx_actual = x_variable.get_gradient()
        dl_db_actual = b_variable.get_gradient()

        np.testing.assert_allclose(dl_dx_actual, dl_dxb)
        np.testing.assert_allclose(dl_db_actual, dl_dxb)

    # self.fail()

    def test_vector(self):
        left = np.array([[1], [2], [3], [4]])
        right = np.array([[5], [6], [7], [8]])

        left_variable = Variable(left)
        right_variable = Variable(right)

        left_right_variable = Add(left_variable, right_variable)

        dl_dleftright = np.array([[9], [10], [11], [12]])

        left_right_variable.forward()
        left_right_variable.backward(grad=dl_dleftright)

        dl_dleft_actual = left_variable.get_gradient()
        dl_dright_actual = right_variable.get_gradient()

        np.testing.assert_allclose(dl_dleft_actual, dl_dleftright)
        np.testing.assert_allclose(dl_dright_actual, dl_dleftright)

        # self.fail()


class TestMSE(TestCase):
    def test_forward(self):
        y_true = np.array([[1], [2], [3], [4]])
        y_predicted = np.array([[8], [7], [6], [5]])
        mse_desired = mean_squared_error(y_true, y_predicted)
        y_true_variable = Variable(y_true)
        y_predicted_variable = Variable(y_predicted)
        mse_node = MSE(y_true_variable, y_predicted_variable)
        mse_actual = mse_node.forward()
        np.testing.assert_allclose(mse_actual, mse_desired)

        # self.fail()

    def test_backward(self):
        y_true = np.array([[1], [2], [3], [4]])
        y_predicted = np.array([[8], [7], [6], [5]])
        mse_derivative_desired = 2.0 / y_true.shape[0] * (y_true - y_predicted)
        y_true_variable = Variable(y_true)
        y_predicted_variable = Variable(y_predicted)
        mse_node = MSE(y_true_variable, y_predicted_variable)
        mse_node.forward()
        mse_node.backward()
        np.testing.assert_allclose(y_true_variable.get_gradient(), mse_derivative_desired)

        # self.fail()


class TestHadamardMult(TestCase):
    def test_forward_backward(self):
        left = np.array([[1], [2], [3], [4]])
        right = np.array([[5], [6], [7], [8]])

        left_variable = Variable(left)
        right_variable = Variable(right)

        left_right_node = HadamardMult(left_variable, right_variable)
        left_right_actual = left_right_node.forward()
        left_right_expected = left * right

        np.testing.assert_allclose(left_right_actual, left_right_expected)

        dl_d_left_right = np.array([[9], [10], [11], [12]])

        left_right_node.backward(dl_d_left_right)

        d_l_d_left_actual = left_variable.get_gradient()
        d_l_d_right_actual = right_variable.get_gradient()

        d_l_d_left_desired = dl_d_left_right * right
        d_l_d_right_desired = dl_d_left_right * left

        np.testing.assert_allclose(d_l_d_left_actual, d_l_d_left_desired)
        np.testing.assert_allclose(d_l_d_right_actual, d_l_d_right_desired)

        # self.fail()

    def test_vector(self):
        left = np.full((4, 1), -1)
        right = np.array([[1], [2], [3], [4]])

        left_variable = Variable(left)
        right_variable = Variable(right)

        left_right_node = HadamardMult(left_variable, right_variable)

        left_right_actual = left_right_node.forward()
        left_right_desired = np.array([[-1], [-2], [-3], [-4]])

        np.testing.assert_allclose(left_right_actual, left_right_desired)

        dl_d_left_right = np.array([[5], [6], [7], [8]])

        left_right_node.backward(dl_d_left_right)

        dl_dleft_actual = left_variable.get_gradient()
        dl_dright_actual = right_variable.get_gradient()
        dl_dleft_desired = np.array([[1 * 5], [2 * 6], [3 * 7], [4 * 8]])
        dl_dright_desired = np.array([[-5], [-6], [-7], [-8]])

        np.testing.assert_allclose(dl_dleft_actual, dl_dleft_desired)
        np.testing.assert_allclose(dl_dright_actual, dl_dright_desired)

        # self.fail()

    def test_square(self):
        x = np.array([[1], [2], [3]])
        x_variable = Variable(x)
        x2_node = HadamardMult(x_variable, x_variable)

        x2_actual = x2_node.forward()
        x2_desired = x * x

        np.testing.assert_almost_equal(x2_actual, x2_desired)

        dl_dx2 = np.array([[4], [5], [6]])

        x2_node.backward(dl_dx2)

        dl_dx_desired = 2.0 * dl_dx2 * x
        dl_dx_actual = x_variable.get_gradient()

        np.testing.assert_almost_equal(dl_dx_desired, dl_dx_actual)


class TestReduceOperations(TestCase):
    def test_reduce_mean_forward_backward(self):
        x = np.array([[1], [2], [3], [4]])

        x_variable = Variable(x)

        reduce_mean_x_node = ReduceMean(x_variable, axis=0)

        reduce_mean_x_actual = reduce_mean_x_node.forward()
        reduce_mean_x_desired = x.mean()

        np.testing.assert_allclose(reduce_mean_x_actual, reduce_mean_x_desired)

        reduce_mean_x_node.backward(grad=np.array([5]))

        dl_dx_actual = x_variable.get_gradient()
        dl_dx_desired = np.full(x.shape, 5.0 / x.shape[0])

        np.testing.assert_allclose(dl_dx_actual, dl_dx_desired)

        # self.fail()

    def test_reduce_mean_merged(self):
        # Array
        y = np.array([-0.5, 1, 2.5])
        v2 = Variable(y)
        m = ReduceMean(v2, 0)
        np.testing.assert_allclose(m.forward(), 1.0, rtol=1e-5)
        m.backward(1)
        np.testing.assert_equal(v2.get_gradient(), [1 / 3, 1 / 3, 1 / 3])

    def test_transpose(self):
        x = np.random.rand(5, 3)
        v = Variable(x)
        t = Transpose(v)
        np.testing.assert_allclose(t.forward(), x.T)
        grads = np.random.rand(3, 5)
        t.backward(grads)
        np.testing.assert_allclose(v.get_gradient(), grads.T)

    def test_reduce_size(self):
        x = np.random.rand(5, 3)
        v = Variable(x)
        rs_full = ReduceSize(v)
        rs_rows = ReduceSize(v, 0)
        rs_cols = ReduceSize(v, 1)
        np.testing.assert_equal(rs_full.forward(), 15)
        np.testing.assert_equal(rs_rows.forward(), 5)
        np.testing.assert_equal(rs_cols.forward(), 3)
        grad_before = v.get_gradient()
        rs_full.backward(np.random.rand(5, 3))
        np.testing.assert_equal(v.get_gradient(), grad_before)

    def test_reduce_sum(self):
        x = np.random.rand(5, 3)
        v1, v2 = Variable(x), Variable(x)
        rs_rows = ReduceSum(v1, 0)
        rs_cols = ReduceSum(v2, 1)
        np.testing.assert_allclose(rs_rows.forward(), np.sum(x, 0))
        np.testing.assert_allclose(rs_cols.forward(), np.sum(x, 1))
        grad = random()
        rs_rows.backward(grad)
        np.testing.assert_allclose(v1.get_gradient(), grad * np.ones((5, 3)))
        rs_cols.backward(grad)
        np.testing.assert_allclose(v2.get_gradient(), grad * np.ones((5, 3)))

    def test_reduce_sum_merged(self):
        # Matrix
        x = np.array([[1, 2, 3], [11, 12, 13]])
        v = Variable(x)
        rs = ReduceSum(v, 1)
        np.testing.assert_allclose(rs.forward(), np.array([6, 36]), rtol=1e-5)
        rs2 = ReduceSum(v, 0)
        np.testing.assert_allclose(rs2.forward(), np.array([12, 14, 16]), rtol=1e-5)
        op_sum = ReduceSum(ReduceSum(v, 0), 0)
        np.testing.assert_allclose(op_sum.forward(), np.sum(x), rtol=1e-5)
        # Array
        y = np.array([-0.5, 1, 2.5])
        v2 = Variable(y)
        r = ReduceSum(v2, 0)
        np.testing.assert_allclose(r.forward(), 3.0, rtol=1e-5)
        r.backward(1)
        np.testing.assert_equal(v2.get_gradient(), [1, 1, 1])


class TestRegularizationMethods(TestCase):
    def test_l1(self):
        x = np.random.rand(50, 30) - 0.5
        v = Variable(x)
        l1 = L1(v)
        np.testing.assert_allclose(l1.forward(), np.sum(np.abs(x)), rtol=1e-5)
        l1.backward(1)
        np.testing.assert_equal(v.get_gradient(), np.sign(x))

    def test_l2(self):
        x = np.random.rand(50, 30) - 0.5
        v = Variable(x)
        l2 = L2(v)
        np.testing.assert_allclose(l2.forward(), np.sum(np.abs(x) ** 2), rtol=1e-5)
        l2.backward(1)
        np.testing.assert_allclose(v.get_gradient(), 2 * x, rtol=1e-5)
