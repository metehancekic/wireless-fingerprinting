import keras
from keras import backend as K
from keras.engine.topology import Layer
from keras.layers import Concatenate

from cxnn.complexnn import ComplexDense, ComplexConv1D, utils

class Modrelu(Layer):

	def __init__(self, **kwargs):
		super(Modrelu, self).__init__(**kwargs)

	def build(self, input_shape):
		# Create a trainable weight variable for this layer.
		self._b = self.add_weight(name='b', 
									shape=(input_shape[-1]//2,),
									initializer='zeros',
									trainable=True)
		super(Modrelu, self).build(input_shape)  # Be sure to call this at the end

	def call(self, x):
		
		real = utils.GetReal()(x)
		imag = utils.GetImag()(x)

		abs1 = K.relu(utils.GetAbs()(x))
		abs2 = K.relu(utils.GetAbs()(x) - self._b)

		
		real = real * abs2 / (abs1+0.0000001)
		imag = imag * abs2 / (abs1+0.0000001)

		merged = Concatenate()([real, imag])

		return merged

	def compute_output_shape(self, input_shape):
		return input_shape