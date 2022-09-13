import tensorflow as tf
from keras import Model, Sequential
from keras.layers import Dense

from Models.Embedding_Model import Embedding
from Models.Transformer_Layer import TransformerLayer

class BERT(Model):
    def __init__(self, i_dim, n_dim, d_dim, h_num, l_num):
        super().__init__()
        '''
        i_dim : item dimension
        n_dim : maximum sequence length
        d_dim : hidden state dimension
        h_num : head number
        l_num : layer number
        '''
        
        self.i_dim = i_dim
        self.n_dim = n_dim
        self.d_dim = d_dim
        
        self.h_num = h_num
        self.l_num = l_num

        # Emb
        self.embedding_layer = Embedding(i_dim, d_dim, n_dim)
        
        # Trm
        self.transformer_seq = Sequential()
        for i in range(l_num):
            self.transformer_seq.add(TransformerLayer(d_dim, h_num))
    
        # Prediction
        self.projection_layer = Dense(d_dim)
        self.prediction_layer = Dense(i_dim)

    def call(self, x):
        x, emb_mat = self.embedding_layer(x)

        x = self.transformer_seq(x)

        x = self.projection_layer(x)
        x = tf.nn.gelu(x)
        
        logits = tf.einsum('b i j , k j -> b i k', x, emb_mat)
        output = tf.nn.softmax(logits, axis = -1)

        return output
