import tensorflow as tf
from keras import Model
from keras.layers import Dense, Softmax, ReLU, LayerNormalization, Dropout
import math

from Models.Blocks.Attention_Block import SelfAttentionBlock, FeedForwardBlock


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





