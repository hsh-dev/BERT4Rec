import tensorflow as tf
from keras import Model
from keras.layers import Dense, Softmax, ReLU, LayerNormalization, Dropout
import math

class FeedForwardBlock(Model):
    def __init__(self, d_dim, scale = 4, activation = "relu"):
        super().__init__()

        self.activation = activation
        self.d_dim = d_dim
        self.linear_layer_1 = Dense(scale * d_dim)
        self.linear_layer_2 = Dense(d_dim)

    def call(self, x):
        x = self.linear_layer_1(x)

        if self.activation == "relu":
            x = tf.nn.relu(x)
        elif self.activation == "gelu":
            x = tf.nn.gelu(x)
            
        x = self.linear_layer_2(x)

        return x
