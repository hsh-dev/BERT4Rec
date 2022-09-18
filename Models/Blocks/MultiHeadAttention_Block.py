import tensorflow as tf
from keras import Model
from keras.layers import Dense

class MHABlock(Model):
    def __init__(self, d_dim, h_num):
        super().__init__()
        
        self.d_dim = d_dim
        self.h_num = h_num
        
        assert d_dim % h_num == 0, "d / h should be integer"
        self.depth = d_dim // h_num
        
        self.w_query = Dense(d_dim)
        self.w_key = Dense(d_dim)
        self.w_value = Dense(d_dim)
        
        self.w_out = Dense(d_dim)
    
    
    def split_head(self, inputs, batch_size):
        inputs = tf.reshape(inputs, shape = (batch_size, -1, self.h_num, self.depth))
        inputs = tf.transpose(inputs, perm = [0, 2, 1, 3])
        return inputs
    
    def attention(self, query, key, value, pad_mask, causality_mask = False):
        '''
        Calculate Attention Score
        B x H x N x D/H -> B x H x N x N
        '''
        dot_product = tf.matmul(query, key, transpose_b=True)
        
        depth = tf.cast(tf.shape(key)[-1], tf.float32)
        logits = dot_product / tf.math.sqrt(depth)
        
        if pad_mask is not None:
            pad_mask = pad_mask[:, tf.newaxis, :, tf.newaxis]
            repeat = tf.constant([1, self.h_num, 1, 1], tf.int32)
            pad_mask = tf.tile(pad_mask, repeat)
            
            pad_mask = tf.cast(tf.math.logical_not(tf.cast(pad_mask, dtype = tf.bool)), dtype = tf.int32)
            product_pad_mask = tf.matmul(pad_mask, pad_mask, transpose_b=True)
            product_pad_mask = tf.cast(tf.math.logical_not(tf.cast(product_pad_mask, dtype = tf.bool)), dtype = tf.float32)
            # product_pad_mask = -product_pad_mask + tf.ones(shape=mask_shape, dtype=tf.int32)
            # product_pad_mask = tf.cast(product_pad_mask, dtype = tf.float32)
            
            mask = product_pad_mask * (-1e6)
            logits = logits + mask

            
        if causality_mask:
            n_dim = tf.shape(logits)[2]
            mask = tf.ones([n_dim, n_dim], dtype=tf.float32)
            mask = tf.linalg.band_part(mask, -1, 0)
            mask = tf.ones([n_dim, n_dim], dtype=tf.float32) - mask
            mask = mask * (-1e20)
            logits = logits + mask
        
        '''
        Softmax
        B x H x N x N
        '''
        attention_weights = tf.nn.softmax(logits, axis=-1)

        '''
        Multiply Value
        B x H x N x D/H
        '''
        output = tf.matmul(attention_weights, value)

        return output

    
    def call(self, x, pad = None):
        '''
        Input : B x N x D
        '''
        batch_size = tf.shape(x)[0]
        
        '''
        Self-Attention
        Shape : B x N x D
        '''
        query = self.w_query(x)
        key = self.w_key(x)
        value = self.w_value(x)
        
        '''
        Split Head
        Shape : B x N x H x D/H -> B x H x N x D/H
        '''
        query = self.split_head(query, batch_size)
        key = self.split_head(key, batch_size)
        value = self.split_head(value, batch_size)
        
        '''
        Calculate Attention Score
        Shape : B x H x N x D/H
        '''
        attention = self.attention(query, key, value, pad_mask = pad, causality_mask = False)
        
        '''
        Concatenate
        Shape : B x H x N x D/H -> B x N x H x D/H -> B x N x D
        '''
        scaled_attention = tf.transpose(attention, perm=[0,2,1,3])
        concat_attention = tf.reshape(scaled_attention, shape = (batch_size, -1, self.d_dim))
        
        
        '''
        Mutltiply Output Matrix
        '''
        output = self.w_out(concat_attention)
        
        return output
