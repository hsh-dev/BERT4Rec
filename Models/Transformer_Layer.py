import tensorflow as tf
from keras import Model
from keras.layers import Dense, Softmax, ReLU, LayerNormalization, Dropout

from Models.Blocks.FeedFoward_Block import FeedForwardBlock
from Models.Blocks.MultiHeadAttention_Block import MHABlock

class TransformerLayer(Model):
    def __init__(self, d_dim, h_num):
        super().__init__()

        self.d_dim = d_dim

        self.layer_norm_1 = LayerNormalization(epsilon=1e-6)
        self.layer_norm_2 = LayerNormalization(epsilon=1e-6)
        self.dropout_1 = Dropout(0.2)
        self.dropout_2 = Dropout(0.2)

        self.mh_att_block = MHABlock(d_dim, h_num)
        
        self.ffn_block = FeedForwardBlock(d_dim, scale = 4, activation = "gelu")

    def call(self, x, pad = None):
        # Multi Head Attention
        shortcut = x
        x = self.mh_att_block(x, pad)
        x = self.dropout_1(x)
        x = x + shortcut
        x = self.layer_norm_1(x)

        # Feed Forward
        shortcut = x
        x = self.ffn_block(x)
        x = self.dropout_2(x)
        x = x + shortcut
        x = self.layer_norm_2(x)
        
        return x
