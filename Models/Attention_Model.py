import tensorflow as tf
from keras import Model
from keras.layers import Dense, Softmax, ReLU, LayerNormalization, Dropout
import math

class SelfAttentionLayer(Model):
    def __init__(self, d_dim):
        super().__init__()

        self.d_dim = d_dim

        self.layer_norm_1 = LayerNormalization(epsilon=1e-6)
        self.layer_norm_2 = LayerNormalization(epsilon=1e-6)
        self.dropout_1 = Dropout(0.2)
        self.dropout_2 = Dropout(0.2)

        self.attention_block = SelfAttentionBlock(d_dim)
        self.ffn_block = FeedForwardBlock(d_dim)

    def call(self, x):
        # Self Attention
        shortcut = x
        x = self.layer_norm_1(x)
        x = self.attention_block(x)
        x = self.dropout_1(x)
        x = x + shortcut

        # Feed Forward
        shortcut = x
        x = self.layer_norm_2(x)
        x = self.ffn_block(x)
        x = self.dropout_2(x)
        x = x + shortcut
        return x




class SelfAttentionBlock(Model):
    def __init__(self, d_dim):
        super().__init__()

        self.d_dim = d_dim

        self.W_q = Dense(d_dim)
        self.W_k = Dense(d_dim)
        self.W_v = Dense(d_dim)

        self.softmax = Softmax(axis=2)

    def call(self, x):
        # Input : B x N x D
        # Query, Key, Value : B x N x D

        query = self.W_q(x)
        key = self.W_k(x)
        value = self.W_v(x)

        # Dot Product : B x N x N
        dot_product = tf.einsum('b i d , b j d -> b i j', query, key)
        dot_product = tf.math.divide(dot_product, float(math.sqrt(self.d_dim)))

        # Causality Mask
        n_dim = dot_product.shape[1]
        mask = tf.ones([n_dim, n_dim], dtype=tf.float32)
        mask = tf.linalg.band_part(mask, -1, 0)
        mask = tf.ones([n_dim, n_dim], dtype=tf.float32) - mask
        mask = mask * (-1e20)

        masked_product = dot_product + mask

        # Calculate Product : B x N x N
        product = self.softmax(masked_product)

        # Dot Product with Value : B x N x D
        attention = tf.einsum('b i j, b j d -> b i d', product, value)

        return attention


class FeedForwardBlock(Model):
    def __init__(self, d_dim):
        super().__init__()

        self.d_dim = d_dim
        self.linear_layer_1 = Dense(d_dim)
        self.relu = ReLU()
        self.linear_layer_2 = Dense(d_dim)

    def call(self, x):
        x = self.linear_layer_1(x)
        x = self.relu(x)
        x = self.linear_layer_2(x)

        return x

