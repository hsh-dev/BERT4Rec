import tensorflow as tf
from tensorflow.keras import Model


class EmbeddingModel(Model):
    def __init__(self, i_dim, d_dim, n_dim):
        super().__init__()

        self.i_dim = i_dim  # total item number

        self.d_dim = d_dim  # latent vector dimension
        self.n_dim = n_dim  # maximum sequence length

        initializer = tf.keras.initializers.GlorotNormal()

        self.item_emb_mat = tf.Variable(
            initializer(shape=[self.i_dim + 2, self.d_dim], dtype=tf.float32),
            trainable=True)

        self.pos_emb_mat = tf.Variable(
            initializer(shape=[self.n_dim, self.d_dim], dtype=tf.float32),
            trainable=True)

    def call(self, x):
        # Input : B x N
        # Emb + Pos : B x N x D
        item_emb = tf.nn.embedding_lookup(self.item_emb_mat, x)
        output = item_emb + self.pos_emb_mat

        emb_matrix = self.item_emb_mat

        return output, emb_matrix
