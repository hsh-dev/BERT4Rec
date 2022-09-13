import tensorflow as tf
from keras import Model
from keras.layers import Dense, Softmax, ReLU, LayerNormalization, Dropout
import math

class SelfAttentionBlock(Model):
    def __init__(self, o_dim, mask=True):
        super().__init__()

        self.o_dim = o_dim

        self.mask = mask

        self.W_q = Dense(o_dim)
        self.W_k = Dense(o_dim)
        self.W_v = Dense(o_dim)

        self.softmax = Softmax(axis=2)

    def call(self, x):
        # Input : B x N x D
        # Query, Key, Value : B x N x D'

        query = self.W_q(x)
        key = self.W_k(x)
        value = self.W_v(x)

        # Dot Product : B x N x N
        dot_product = tf.einsum('b i d , b j d -> b i j', query, key)
        dot_product = tf.math.divide(dot_product, float(math.sqrt(self.o_dim)))

        if self.mask:
            # Causality Mask
            n_dim = dot_product.shape[1]
            mask = tf.ones([n_dim, n_dim], dtype=tf.float32)
            mask = tf.linalg.band_part(mask, -1, 0)
            mask = tf.ones([n_dim, n_dim], dtype=tf.float32) - mask
            mask = mask * (-1e20)
            dot_product = dot_product + mask

        # Calculate Product : B x N x N
        product = self.softmax(dot_product)

        # Dot Product with Value : B x N x D
        attention = tf.einsum('b i j, b j d -> b i d', product, value)

        return attention

