import numpy as np
from keras import backend as K
from keras.engine.topology import Layer
from keras.layers import Reshape


class RepeatLayer(Layer):
    def __init__(self, repeat, axis, **kwargs):
        super(RepeatLayer, self).__init__(**kwargs)
        self._axis = RepeatLayer.fix_axes_considering_batch_dimention(axis)
        self._repeat = repeat

    # def build(self, input_shape):
    #     super(RepeatLayer, self).build(input_shape)

    @staticmethod
    def fix_axes_considering_batch_dimention(axis):
        if axis >= 0:
            return 1 + axis
        else:
            return axis

    def call(self, inputs, **kwargs):
        return K.repeat_elements(inputs, self._repeat, self._axis)

    def compute_output_shape(self, input_shape):
        output_shape = np.copy(input_shape)
        output_shape[self._axis] *= self._repeat

        return output_shape


class ExpandDimension(Layer):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    # def build(self, input_shape):
    #     super(AddDimension, self).build(input_shape)

    def compute_output_shape(self, input_shape):
        output_shape = np.append(input_shape, 1)

        return output_shape

    def call(self, inputs, **kwargs):
        outputs = K.expand_dims(inputs)
        return outputs

