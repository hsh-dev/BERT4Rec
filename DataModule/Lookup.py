import tensorflow as tf


class StringLookup():
    def __init__(self, items) -> None:
        self.items = items  # item keys init
        self.items.append("mask")
        
        self.encoding_layer = tf.keras.layers.StringLookup(
            vocabulary=self.items)

        self.decoding_layer = tf.keras.layers.StringLookup(
            vocabulary=self.items, invert=True)

    def str_to_idx(self, x):
        return self.encoding_layer(x)

    def idx_to_str(self, x):
        decoded_x = self.decoding_layer(x)
        list_x = decoded_x.numpy().tolist()
        str_x = list(map(lambda x: x.decode('UTF-8'), list_x))

        return str_x

    # def byte_to_str(self, x):
    #     # list_x = x.numpy().tolist()
    #     str_x = list(map(lambda x: x.decode('UTF-8'), list_x))
        
    #     return str_x
