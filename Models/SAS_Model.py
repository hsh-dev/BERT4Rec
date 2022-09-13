import tensorflow as tf
from tensorflow.keras import Model, Sequential
from tensorflow.keras.layers import Dense

from Models.Embedding_Model import Embedding
from Models.Attention_Layer import SelfAttentionLayer

class SAS(Model):
    def __init__(self, i_dim, n_dim, d_dim, b_num):
        super().__init__()

        self.i_dim = i_dim
        self.n_dim = n_dim
        self.d_dim = d_dim
        self.b_num = b_num

        # Emb
        self.embedding_layer = Embedding(i_dim, d_dim, n_dim)

        # SA + FFN
        self.sequential_attention = Sequential()

        for i in range(b_num):
            self.sequential_attention.add(SelfAttentionLayer(d_dim))

        # Prediction
        self.prediction_layer = Dense(i_dim)

    def call(self, x):
        x, emb_mat = self.embedding_layer(x)

        x = self.sequential_attention(x)

        output = tf.einsum('b i j , k j -> b i k', x, emb_mat)

        return output
