from random import random, seed
from unittest import TestCase

from graph.UnaryOperations import ReduceMean, ReduceSum, ReduceSize, Transpose, Log
from graph.Variable import Variable
from graph.BinaryOperations import Add, Multiply, HadamardMult
from mydnn import mydnn
from utils.LossFunctions import MSE, MSEWithSplitter, CrossEntropy
import numpy as np
from sklearn.metrics import mean_squared_error, log_loss
from utils.RegularizationMethods import L1, L2
from FullyConnectedLayer import FullyConnectedLayer
from utils.ActivationFunctions import Identity, Softmax
import tensorflow as tf
from keras import layers, optimizers, Sequential, regularizers, losses, metrics


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

    def test_reduce_mean_broadcasting(self):
        x = np.arange(6).reshape(3, 2)
        w = np.arange(6, 8).reshape(2, 1)
        b = 12.0
        y = np.arange(8, 11).reshape(3, 1)
        dl_mse = 11.0

        x_variable = Variable(x)
        w_variable = Variable(w)
        b_variable = Variable(b)
        y_variable = Variable(y)

        xw_node = Multiply(x_variable, w_variable)
        xwb_node = Add(xw_node, b_variable)
        xwb_mse_node = MSE(y_variable, xwb_node)

        xwb_mse_desired = mean_squared_error(y, (x @ w) + np.full((3, 1), b))
        xwb_mean_actual = xwb_mse_node.forward()
        np.testing.assert_allclose(xwb_mean_actual, xwb_mse_desired)
        xwb_mse_node.backward(dl_mse)

        dl_db_actual = b_variable.get_gradient()
        dl_db_desired = dl_mse * 2.0 * np.sum((x @ w) + np.full((3, 1), b) - y) / x.shape[0]

        np.testing.assert_allclose(dl_db_actual, dl_db_desired)

        dl_dx = x_variable.get_gradient()
        dl_dw = w_variable.get_gradient()

    def test_splitter_forward(self):
        y_true = np.array([[1], [2], [3], [4]])
        y_predicted = np.array([[8], [7], [6], [5]])
        mse_desired = mean_squared_error(y_true, y_predicted)
        y_true_variable = Variable(y_true)
        y_predicted_variable = Variable(y_predicted)
        mse_node = MSEWithSplitter(y_true_variable, y_predicted_variable)
        mse_actual = mse_node.forward()
        np.testing.assert_allclose(mse_actual, mse_desired)

    def test_splitter_backward(self):
        y_true = np.array([[1], [2], [3], [4]])
        y_predicted = np.array([[8], [7], [6], [5]])
        mse_derivative_desired = 2.0 / y_true.shape[0] * (y_true - y_predicted)
        y_true_variable = Variable(y_true)
        y_predicted_variable = Variable(y_predicted)
        mse_node = MSEWithSplitter(y_true_variable, y_predicted_variable)
        mse_node.forward()
        mse_node.backward()
        np.testing.assert_allclose(y_true_variable.get_gradient(), mse_derivative_desired)

    def test_reduce_mean_splitter_broadcasting(self):
        x = np.arange(6).reshape(3, 2)
        w = np.arange(6, 8).reshape(2, 1)
        b = 12.0
        y = np.arange(8, 11).reshape(3, 1)
        dl_mse = 11.0

        x_variable = Variable(x)
        w_variable = Variable(w)
        b_variable = Variable(b)
        y_variable = Variable(y)

        xw_node = Multiply(x_variable, w_variable)
        xwb_node = Add(xw_node, b_variable)
        xwb_mse_node = MSEWithSplitter(y_variable, xwb_node)

        xwb_mse_desired = mean_squared_error(y, (x @ w) + np.full((3, 1), b))
        xwb_mean_actual = xwb_mse_node.forward()
        np.testing.assert_allclose(xwb_mean_actual, xwb_mse_desired)
        xwb_mse_node.backward(dl_mse)

        dl_db_actual = b_variable.get_gradient()
        dl_db_desired = dl_mse * 2.0 * np.sum((x @ w) + np.full((3, 1), b) - y) / x.shape[0]

        np.testing.assert_allclose(dl_db_actual, dl_db_desired)

        dl_dx = x_variable.get_gradient()
        dl_dw = w_variable.get_gradient()


class TestCE(TestCase):
    def test_forward(self):
        y_true = np.array([[0.0, 1.0],
                           [1.0, 0.0],
                           [0.0, 1.0]])
        y_predicted = np.arange(1, 7).reshape(3, 2)
        y_predicted = y_predicted / np.tile(np.sum(y_predicted, axis=1).reshape(3, 1), (1,2))

        y_true_variable = Variable(y_true)
        y_predicted_variable = Variable(y_predicted)

        ce_node = CrossEntropy(y_true_variable, y_predicted_variable)

        ce_actual = ce_node.forward()
        ce_desired = log_loss(y_true, y_predicted)

        self.assertAlmostEqual(ce_actual, ce_desired)

        dl_dce = 2.0

        ce_node.backward(dl_dce)

        dl_dyp_actual = y_predicted_variable.get_gradient()
        dl_dyp_desired = -dl_dce * (y_true / y_predicted) / y_true.shape[0]

        np.testing.assert_allclose(dl_dyp_actual, dl_dyp_desired)


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


class TestLog(TestCase):
    def test_forward_backward(self):
        y = np.arange(1, 7).reshape(3, 2)
        y_variable = Variable(y)

        log_node = Log(y_variable)

        y_log_actual = log_node.forward()
        y_log_desired = np.log(y)

        np.testing.assert_allclose(y_log_actual, y_log_desired)

        dl_dlogy = np.arange(7, 13).reshape(3, 2)

        log_node.backward(dl_dlogy)

        dl_dy_actual = y_variable.get_gradient()
        dl_dy_desired = dl_dlogy / y

        np.testing.assert_allclose(dl_dy_actual, dl_dy_desired)


class TestReduceOperations(TestCase):
    def test_reduce_mean_forward_backward(self):
        x = np.array([[1.0], [2.0], [3.0], [4.0]])

        x_variable = Variable(x)

        reduce_mean_x_node = ReduceMean(x_variable, axis=0)

        reduce_mean_x_actual = reduce_mean_x_node.forward()
        reduce_mean_x_desired = x.mean()

        np.testing.assert_allclose(reduce_mean_x_actual, reduce_mean_x_desired)

        reduce_mean_x_node.backward(grad=np.array([5.0]))

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
        m.backward(1.0)
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
        grad_rows = np.random.rand(3,)
        rs_rows.backward(grad_rows)
        np.testing.assert_allclose(v1.get_gradient(), grad_rows * np.ones((5, 3)))
        grad_cols = np.random.rand(5,)
        rs_cols.backward(grad_cols)
        np.testing.assert_allclose(v2.get_gradient(), (grad_cols * np.ones((5, 3)).T).T)

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


class TestFC(TestCase):
    def test_forward_backward_1_no_activation(self):
        x = np.arange(6).reshape(3, 2)
        x_variable = Variable(x)
        fc = FullyConnectedLayer(2, 1, Identity, x_variable)
        w = fc._w._value.copy()
        b = fc._b._value.copy()
        wxb_desired = x @ w + b
        wxb_actual = fc.forward()

        np.testing.assert_almost_equal(wxb_actual, wxb_desired)

        fc.backward(np.array([[6.0], [7.0], [8.0]]))

        dl_dw_actual = fc._w.get_gradient()
        dl_dx_actual = x_variable.get_gradient()
        dl_dw_desired = np.array([[0 * 6 + 2 * 7 + 4 * 8], [1 * 6 + 3 * 7 + 5 * 8]])
        dl_dx_desired = np.array([[w[0,0] * 6, w[1,0] * 6], [w[0,0] * 7, w[1,0] * 7], [w[0,0] * 8, w[1,0] * 8]])

        np.testing.assert_allclose(dl_dw_actual, dl_dw_desired)
        np.testing.assert_allclose(dl_dx_actual, dl_dx_desired)

        dl_db_actual = fc._b.get_gradient()
        dl_db_desired = np.array([6 + 7 + 8])

        np.testing.assert_allclose(dl_db_actual, dl_db_desired)

    def test_forward_backward_4_no_activation(self):
        x = np.arange(6).reshape(3, 2)
        x_variable = Variable(x)
        fc = FullyConnectedLayer(2, 4, Identity, x_variable)
        w = fc._w._value.copy()
        b = fc._b._value.copy()
        wxb_desired = x @ w + b
        wxb_actual = fc.forward()

        np.testing.assert_almost_equal(wxb_actual, wxb_desired)

        dl_dxwb = np.arange(6, 6 + 3 * 4).reshape(3, 4)

        fc.backward(dl_dxwb)

        dl_dw_actual = fc._w.get_gradient()
        dl_dw_desired = np.array([
            [x[:, 0].T @ dl_dxwb[:, 0], x[:, 0].T @ dl_dxwb[:, 1], x[:, 0].T @ dl_dxwb[:, 2],
             x[:, 0].T @ dl_dxwb[:, 3]],
            [x[:, 1].T @ dl_dxwb[:, 0], x[:, 1].T @ dl_dxwb[:, 1], x[:, 1].T @ dl_dxwb[:, 2],
             x[:, 1].T @ dl_dxwb[:, 3]],
        ])

        np.testing.assert_allclose(dl_dw_actual, dl_dw_desired)

        # self.fail()


class TestSoftMax(TestCase):
    def test_forward_backward(self):
        np.random.seed(42)
        y = np.random.rand(3, 2)
        y_variable = Variable(y)
        softmax_node = Softmax(y_variable)
        y_softmax_actual = softmax_node.forward()
        ey = np.exp(y)
        y_softmax_desired = (ey.T / np.sum(ey, axis=1)).T
        np.testing.assert_allclose(y_softmax_actual, y_softmax_desired)

        dl_dsoftmax = np.random.rand(3,2)

        # ---------------
        # | 6*e^0| 7*e^1|
        # ---------------
        # | 8*e^2| 9*e^3|
        # ---------------
        # |10*e^4|11*e^5|
        # ---------------
        weighted_ey = dl_dsoftmax * ey
        # ----------
        # | e^0+e^1|
        # ----------
        # | e^2+e^3|
        # ----------
        # | e^4+e^5|
        # ----------
        ey_row_sum = ey.sum(axis=1)
        # ----------------
        # |  6*e^0+ 7*e^1|
        # ----------------
        # |  8*e^2+ 8*e^3|
        # ----------------
        # | 10*e^4+11*e^5|
        # ----------------
        weighted_ey_row_sum = weighted_ey.sum(axis=1)
        # --------------
        # | (e^0+e^1)^2|
        # --------------
        # | (e^2+e^3)^2|
        # --------------
        # | (e^4+e^5)^2|
        # --------------
        squared_ey_row_sum = np.square(ey_row_sum)

        dl_dy_desired = np.array([
            [(weighted_ey[0,0] * ey_row_sum[0] - ey[0,0] * weighted_ey_row_sum[0]) / squared_ey_row_sum[0], (weighted_ey[0,1] * ey_row_sum[0] - ey[0,1] * weighted_ey_row_sum[0]) / squared_ey_row_sum[0]],
            [(weighted_ey[1,0] * ey_row_sum[1] - ey[1,0] * weighted_ey_row_sum[1]) / squared_ey_row_sum[1], (weighted_ey[1,1] * ey_row_sum[1] - ey[1,1] * weighted_ey_row_sum[1]) / squared_ey_row_sum[1]],
            [(weighted_ey[2,0] * ey_row_sum[2] - ey[2,0] * weighted_ey_row_sum[2]) / squared_ey_row_sum[2], (weighted_ey[2,1] * ey_row_sum[2] - ey[2,1] * weighted_ey_row_sum[2]) / squared_ey_row_sum[2]],
        ])

        softmax_node.backward(dl_dsoftmax)

        dl_dy_actual = y_variable.get_gradient()

        np.testing.assert_allclose(dl_dy_actual, dl_dy_desired)


class TestMyDNN(TestCase):
    def test_one_layer_none_activation_none_regularization_mse(self):
        np.random.seed(42)

        actual = mydnn([
            {
                'input': 2,
                'output': 4,
                'nonlinear': 'none',
                'regularization': 'l2',
            },
            {
                'input': 4,
                'output': 1,
                'nonlinear': 'none',
                'regularization': 'l2',
            }
        ], 'MSE')

        x = np.arange(6).reshape(3, 2)
        y = np.array([[6], [7], [8]])

        w1_before = actual._architecture[0]._w._value.copy()

        def kernel_initializer1(shape, dtype=None):
            w = w1_before
            assert w.shape == shape

            return w

        b1_before = actual._architecture[0]._b._value.copy()

        def bias_initializer1(shape, dtype=None):
            b = b1_before
            assert b.shape == shape

            return b

        w2_before = actual._architecture[1]._w._value.copy()

        def kernel_initializer2(shape, dtype=None):
            w = w2_before
            assert w.shape == shape

            return w

        b2_before = actual._architecture[1]._b._value.copy()

        def bias_initializer2(shape, dtype=None):
            b = b2_before
            assert b.shape == shape

            return b

        actual.fit(x, y, 1, x.shape[0], 0.01)

        b1_actual = actual._architecture[0]._b._value


        # np.random.seed(42)

        # dl_db_desired = 2.0 * np.sum(x @ w_before + np.full((3, 1), b_before) - y) / x.shape[0]
        # dl_db_actual1 = actual._architecture[0]._b.get_gradient()
        # dl_db_actual2 = (b_before - b_actual) / 0.01
        #
        # np.testing.assert_allclose(dl_db_actual1, dl_db_desired)
        # np.testing.assert_allclose(dl_db_actual2, dl_db_desired)

        y_predictions = ((((x @ w1_before) + b1_before) @ w2_before) + b2_before)
        dl_db_desired = 2.0 * np.sum((y_predictions - y) @ np.transpose(w2_before), axis=0) / x.shape[0]
        db_db_actual = (b1_before - b1_actual) / 0.01

        np.testing.assert_allclose(db_db_actual, dl_db_desired)

    def test_two_layers_two_hidden_units_none_activation_none_regularization_mse(self):

        np.random.seed(42)

        actual = mydnn([
            {
                'input': 2,
                'output': 1,
                'nonlinear': 'none',
                'regularization': 'l2',
            }
        ], 'MSE')

        x = np.arange(6).reshape(3, 2)
        y = np.array([[6], [7], [8]])

        w_before = actual._architecture[0]._w._value.copy()

        def kernel_initializer(shape, dtype=None):
            w = w_before
            assert w.shape == shape

            return w

        b_before = actual._architecture[0]._b._value.copy()

        def bias_initializer(shape, dtype=None):
            b = b_before
            assert b.shape == shape

            return b

        sgd = optimizers.SGD()
        desired = Sequential()
        desired.add(layers.Dense(1, kernel_initializer=kernel_initializer, bias_initializer=bias_initializer,
                                 input_shape=(x.shape[1],)))
        desired.compile(sgd, 'MSE')

        desired.fit(x, y, batch_size=x.shape[0])
        actual.fit(x, y, 1, x.shape[0], 0.01)

        w_desired, b_desired = desired.get_weights()
        w_actual = actual._architecture[0]._w._value
        b_actual = actual._architecture[0]._b._value

        np.random.seed(42)

        dl_db_desired = 2.0 * np.sum(x @ w_before + np.full((3,1), b_before) - y) / x.shape[0]
        dl_db_actual2 = (b_before - b_actual) / 0.01

        np.testing.assert_allclose(dl_db_actual2, dl_db_desired)

        np.testing.assert_allclose(b_actual, b_desired)
        np.testing.assert_allclose(w_actual, w_desired)

    def test_one_layer_relu_activation_none_regularization_mse(self):
        np.random.seed(42)

        actual = mydnn([
            {
                'input': 2,
                'output': 1,
                'nonlinear': 'relu',
                'regularization': 'l2',
            }
        ], 'MSE')

        x = np.random.normal(size=(3, 2))
        y = np.random.normal(size=(3,1))

        w_before = actual._architecture[0]._w._value.copy()

        def kernel_initializer(shape, dtype=None):
            w = w_before
            assert w.shape == shape

            return w

        b_before = actual._architecture[0]._b._value.copy()

        def bias_initializer(shape, dtype=None):
            b = b_before
            assert b.shape == shape

            return b

        sgd = optimizers.SGD()
        desired = Sequential()
        desired.add(layers.Dense(1, activation='relu',
                                 kernel_initializer=kernel_initializer, bias_initializer=bias_initializer,
                                 input_shape=(x.shape[1],)))
        desired.compile(sgd, 'MSE')

        desired.fit(x, y, batch_size=x.shape[0])
        actual.fit(x, y, 1, x.shape[0], 0.01)

        w_desired, b_desired = desired.get_weights()
        w_actual = actual._architecture[0]._w._value
        b_actual = actual._architecture[0]._b._value

        np.random.seed(42)

        #dl_db_desired = 2.0 * np.sum(x @ w_before + np.full((3,1), b_before) - y) / x.shape[0]
        dl_db_actual1 = actual._architecture[0]._b.get_gradient()
        dl_db_actual2 = (b_before - b_actual) / 0.01

        # np.testing.assert_allclose(dl_db_actual1, dl_db_desired)
        # np.testing.assert_allclose(dl_db_actual2, dl_db_desired)

        np.testing.assert_allclose(b_actual, b_desired)
        np.testing.assert_allclose(w_actual, w_desired)

    def test_one_layer_sigmoid_activation_none_regularization_mse(self):
        np.random.seed(42)

        actual = mydnn([
            {
                'input': 2,
                'output': 1,
                'nonlinear': 'sigmoid',
                'regularization': 'l2',
            }
        ], 'MSE')

        x = np.random.normal(size=(3, 2))
        y = np.random.normal(size=(3,1))

        w_before = actual._architecture[0]._w._value.copy()

        def kernel_initializer(shape, dtype=None):
            w = w_before
            assert w.shape == shape

            return w

        b_before = actual._architecture[0]._b._value.copy()

        def bias_initializer(shape, dtype=None):
            b = b_before
            assert b.shape == shape

            return b

        sgd = optimizers.SGD()
        desired = Sequential()
        desired.add(layers.Dense(1, activation='sigmoid',
                                 kernel_initializer=kernel_initializer, bias_initializer=bias_initializer,
                                 input_shape=(x.shape[1],)))
        desired.compile(sgd, 'MSE')

        desired.fit(x, y, batch_size=x.shape[0])
        actual.fit(x, y, 1, x.shape[0], 0.01)

        w_desired, b_desired = desired.get_weights()
        w_actual = actual._architecture[0]._w._value
        b_actual = actual._architecture[0]._b._value

        np.random.seed(42)

        np.testing.assert_allclose(b_actual, b_desired, rtol=1e-7, atol=1e-9)
        np.testing.assert_allclose(w_actual, w_desired, rtol=1e-7, atol=1e-9)

    def test_one_layer_softmax_activation_none_regularization_mse(self):
        TestMyDNN._compare_models([{'input': 4, 'output': 3, 'nonlinear': 'sot-max', 'regularization': 'l2'}], 'cross-entropy')

    def test_one_layer_none_activation_l2_regularization_mse(self):
        np.random.seed(42)

        weight_decay = 10.0

        actual = mydnn([
            {
                'input': 2,
                'output': 1,
                'nonlinear': 'none',
                'regularization': 'l2',
            }
        ], 'MSE', weight_decay=weight_decay)

        x = np.arange(6).reshape(3, 2)
        y = np.array([[6], [7], [8]])

        w_before = actual._architecture[0]._w._value.copy()

        def kernel_initializer(shape, dtype=None):
            w = w_before
            assert w.shape == shape

            return w

        b_before = actual._architecture[0]._b._value.copy()

        def bias_initializer(shape, dtype=None):
            b = b_before
            assert b.shape == shape

            return b

        sgd = optimizers.SGD()
        desired = Sequential()
        desired.add(layers.Dense(1, kernel_initializer=kernel_initializer, bias_initializer=bias_initializer,
                                 input_shape=(x.shape[1],), kernel_regularizer=regularizers.l2(weight_decay)))
        desired.compile(sgd, 'MSE')

        desired.fit(x, y, batch_size=x.shape[0])
        actual.fit(x, y, 1, x.shape[0], 0.01)

        w_desired, b_desired = desired.get_weights()
        w_actual = actual._architecture[0]._w._value
        b_actual = actual._architecture[0]._b._value

        dl_dw_desired2 = 2.0 * (np.transpose(x) @ (x @ w_before + b_before - y) / x.shape[0] + weight_decay * w_before)
        dl_dw_desired1 = (w_before - w_desired) / 0.01
        dl_dw_actual1 = (w_before - w_actual) / 0.01

        np.testing.assert_allclose(dl_dw_actual1, dl_dw_desired1, atol=1e-5)
        np.testing.assert_allclose(dl_dw_actual1, dl_dw_desired2, atol=1e-5)
        np.testing.assert_allclose(b_actual, b_desired)
        np.testing.assert_allclose(w_actual, w_desired, atol=1e-5)

    def test_one_layer_none_activation_l1_regularization_mse(self):
        np.random.seed(42)

        weight_decay = 10.0

        actual = mydnn([
            {
                'input': 2,
                'output': 1,
                'nonlinear': 'none',
                'regularization': 'l1',
            }
        ], 'MSE', weight_decay=weight_decay)

        x = np.arange(6).reshape(3, 2)
        y = np.array([[6], [7], [8]])

        w_before = actual._architecture[0]._w._value.copy()

        def kernel_initializer(shape, dtype=None):
            w = w_before
            assert w.shape == shape

            return w

        b_before = actual._architecture[0]._b._value.copy()

        def bias_initializer(shape, dtype=None):
            b = b_before
            assert b.shape == shape

            return b

        sgd = optimizers.SGD()
        desired = Sequential()
        desired.add(layers.Dense(1, kernel_initializer=kernel_initializer, bias_initializer=bias_initializer,
                                 input_shape=(x.shape[1],), kernel_regularizer=regularizers.l1(weight_decay)))
        desired.compile(sgd, 'MSE')

        desired.fit(x, y, batch_size=x.shape[0])
        actual.fit(x, y, 1, x.shape[0], 0.01)

        w_desired, b_desired = desired.get_weights()
        w_actual = actual._architecture[0]._w._value
        b_actual = actual._architecture[0]._b._value

        dl_dw_desired1 = (w_before - w_desired) / 0.01
        dl_dw_actual1 = (w_before - w_actual) / 0.01
        dl_dw_actual2 = actual._architecture[0]._w.get_gradient()

        np.testing.assert_allclose(dl_dw_actual1, dl_dw_desired1, atol=1e-5)
        np.testing.assert_allclose(b_actual, b_desired)
        np.testing.assert_allclose(w_actual, w_desired, atol=1e-5)

    #TODO: use this function in previous cases
    @staticmethod
    def _compare_models(architecture, loss, weight_decay=0.0, number_of_samples=3, learning_rate=0.01, epochs=1):
        np.random.seed(42)

        x = np.random.rand(number_of_samples, architecture[0]['input'])
        if 'MSE' == loss:
            y = np.random.rand(number_of_samples, 1)
            keras_loss = 'MSE'
            keras_metrics = None
        elif 'cross-entropy' == loss:
            number_of_classes = architecture[-1]['output']
            y = np.eye(number_of_classes)[np.random.choice(number_of_classes, size=number_of_samples)]
            keras_loss = losses.categorical_crossentropy
            keras_metrics = [metrics.categorical_accuracy]
        else:
            assert False, loss

        actual = mydnn(architecture,
                       loss,
                       weight_decay)
        sgd = optimizers.SGD(lr=learning_rate)
        desired = Sequential()

        for index, layer in enumerate(architecture):
            units = layer['output']
            activation = layer['nonlinear']

            if 'sot-max' == activation:
                activation = 'softmax'

            def kernel_initializer(shape, dtype=None):
                w = actual._architecture[index]._w.get_value()
                assert w.shape == shape

                return w

            def bias_initializer(shape, dtype=None):
                b = actual._architecture[index]._b.get_value()
                assert b.shape == shape

                return b

            if 'l1' == layer['regularization']:
                kernel_regularizer = regularizers.l1(weight_decay)
            elif 'l2' == layer['regularization']:
                kernel_regularizer = regularizers.l2(weight_decay)
            else:
                assert False

            desired.add(layers.Dense(units,
                                     activation=activation,
                                     kernel_initializer=kernel_initializer,
                                     bias_initializer=bias_initializer,
                                     kernel_regularizer=kernel_regularizer,
                                     input_shape=(layer['input'],)))

        desired.compile(sgd, keras_loss, metrics=keras_metrics)

        desired.fit(x, y, batch_size=x.shape[0], epochs=1)
        actual.fit(x, y, epochs, x.shape[0], learning_rate)

        weights_and_biases_desired = desired.get_weights()
        weights_and_biases_actual = list()

        for index in range(len(architecture)):
            weights_and_biases_actual.append(actual._architecture[index]._w.get_value())
            weights_and_biases_actual.append(actual._architecture[index]._b.get_value())

        for weight_actual, weight_desired in zip(weights_and_biases_actual, weights_and_biases_desired):
            np.testing.assert_allclose(weight_actual, weight_desired, atol=1e-5)

        y_predicted_actual = actual.predict(x, batch_size=number_of_samples)
        y_predicted_desired = desired.predict(x, batch_size=number_of_samples)

        np.testing.assert_allclose(y_predicted_actual, y_predicted_desired)

        if 'MSE' == loss:
            loss_actual = actual.evaluate(x, y, batch_size=number_of_samples)
            loss_desired = desired.evaluate(x, y, batch_size=number_of_samples)

            np.testing.assert_allclose(loss_actual, loss_desired)
        else:
            loss_actual, accuracy_actual = actual.evaluate(x, y, batch_size=number_of_samples)
            loss_desired, accuracy_desired = desired.evaluate(x, y, batch_size=number_of_samples)

            np.testing.assert_allclose(loss_actual, loss_desired)
            np.testing.assert_allclose(accuracy_actual, accuracy_desired)

    def test_1_layer_relu_no_regularization_mse_2_epochs(self):
        TestMyDNN._compare_models([
            {
                'input': 4,
                'output': 1,
                'nonlinear': 'relu',
                'regularization': 'l2'
            }
                ],
            'MSE',
            epochs=2
        )

    # TODO: check it is precision problem and not some bloody bug
    # def test_1_layer_sm_no_regularization_ce_2_epochs(self):
    #     TestMyDNN._compare_models([
    #         {
    #             'input': 4,
    #             'output': 3,
    #             'nonlinear': 'sot-max',
    #             'regularization': 'l2'
    #         }
    #             ],
    #         'cross-entropy',
    #         epochs=2
    #     )

    # TODO: check it is precision problem and not some bloody bug
    # def test_2_layers_nelu_soft_max_none_regularization_ce_2_epochs(self):
    #     TestMyDNN._compare_models([
    #         {
    #             'input': 4,
    #             'output': 3,
    #             'nonlinear': 'relu',
    #             'regularization': 'l2'
    #         },
    #         {
    #             'input': 3,
    #             'output': 5,
    #             'nonlinear': 'sot-max',
    #             'regularization': 'l2'
    #         }
    #     ],
    #         'cross-entropy',
    #         epochs=2
    #     )


if "__main__" == __name__:
    TestSoftMax().test_forward_backward()
    TestMyDNN().test_one_layer_sigmoid_activation_none_regularization_mse()
    TestReduceOperations().test_reduce_sum()
    TestCE().test_forward()

    import sys
    import inspect

    current_module = sys.modules[__name__]

    for class_name, class_object in inspect.getmembers(sys.modules[__name__], predicate=inspect.isclass):
        if 'Test' == class_name[:len('Test')]:
            test_object = class_object()

            for function_name, function_object in inspect.getmembers(test_object, predicate=inspect.ismethod):
                if 'test' in function_name:
                    function_object()
