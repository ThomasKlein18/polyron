
import tensorflow as tf 
import numpy as np
from tensorflow.keras.layers import Dense 
from tensorflow.keras.layers import Layer 

tanh_coefficients = [0,1,0,-0.333, 0, 2/15, 0, -17/315, 0, 62/2835]

class Polyron(Layer):

    def __init__(self, units, degree, initializer=None, **kwargs):
        """

        """

        super(Polyron, self).__init__(**kwargs)

        self.units = units
        self.degree = degree
        if initializer == "linear":
            init_values = np.zeros((units, degree))
            init_values[:,1] = 1 # f(x) = x, might work better than random, because closer to relu
            self.initializer = tf.keras.initializers.constant(init_values)
        elif initializer == "tanh":
            init_values = np.zeros((units, degree))
            for i in range(degree):
                init_values[:,i] = tanh_coefficients[i] # trying to approximate tanh
            self.initializer = tf.keras.initializers.constant(init_values)
        else:
            self.initializer = initializer
        self.dense_layer = Dense(units)


    def build(self, input_shape):

        self.coefficients = self.add_weight(
            shape=[self.units, self.degree],
            name='coefficients',
            initializer=self.initializer)
            
        self.built = True

    def call(self, inputs):
        """
        I think inputs will have shape batchsize x neurons. 
        """
        linear = self.dense_layer(inputs)
        exp = tf.stack([tf.math.pow(linear,i) for i in range(self.degree)])
        res = tf.stack([[exp[j,:,i] * self.coefficients[i,j] for i in range(self.units)] for j in range(self.degree)]) #sorry

        return tf.transpose(tf.reduce_sum(res, axis=0))

    def get_config(self):
        config = {
            'degree': self.degree
        }
        base_config = super(Polyron, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

    def compute_output_shape(self, input_shape):
        return input_shape

