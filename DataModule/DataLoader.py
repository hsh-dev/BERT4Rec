import pandas as pd
import numpy as np
import tensorflow as tf
import random
from tqdm import tqdm

from DataModule.Lookup import StringLookup
import time

class DataLoader():
    def __init__(self, config) -> None:
        self.config = config
        
        self.batch_size = config["batch_size"]
        self.split_ratio = config["split_ratio"]
        
        self.movies_path = config["movies_path"]
        self.ratings_path = config["ratings_path"]
        self.users_path = config["users_path"]
        
        self.negative_sample_count = config["negative_sample_count"]
        
        self.negative_sample = {}        
        
        self._load_()
        self._init_data_()
        self._init_length_()
        self._init_string_lookup_()
        self._make_session_()
        
    
    def _load_(self):
        self.movies_data = pd.read_csv(self.movies_path, delimiter = "::", header = None, engine = "python", encoding = "ISO-8859-1")
        self.ratings_data = pd.read_csv(self.ratings_path, delimiter = "::", header = None, engine = "python", encoding = "ISO-8859-1")
        self.users_data = pd.read_csv(self.users_path, delimiter = "::", header = None, engine = "python", encoding = "ISO-8859-1")

    def _init_length_(self):
        self.movie_len = len(self.movie_ids)
    
    def _init_data_(self):
        self.movie_ids = list(
                        map(lambda x : str(x), self.movies_data[0].unique()))
        self.user_ids = list(
                        map(lambda x: str(x), self.users_data[0].unique()))
    
    def _init_string_lookup_(self):
        item_list = self.get_movie_ids()
        self.string_lookup = StringLookup(item_list)
        
    
    def _make_session_(self):
        train_user, valid_user = self.split_user()
        
        train_set = self.collect_movie_list(train_user)
        valid_set = self.collect_movie_list(valid_user)   
        
        self._make_negative_sample(train_set, valid_set)
        
        # self.train_x, self.train_y, self.train_u = self.make_seq_to_seq(train_set, 20)              
        # self.valid_x, self.valid_y, self.valid_u = self.make_seq_to_seq(valid_set, 20)        
        self.train_x, self.train_y, self.train_m, self.train_p = self.make_mask_seq(train_set, "train")
        self.valid_x, self.valid_y, self.valid_m, self.valid_p = self.make_mask_seq(valid_set, "valid")


    def _make_negative_sample(self, train_set, valid_set):
        self.negative_sample = {}        
        keys = train_set.keys()
        
        for key in keys:
            movie_list = train_set[key]
            comp_list = list(set(self.movie_ids) - set(movie_list))
            
            negative_item_list = random.sample(comp_list, self.negative_sample_count)
            self.negative_sample[str(key)] = negative_item_list
        
        keys = valid_set.keys()
        for key in keys:
            movie_list = valid_set[key]
            comp_list = list(set(self.movie_ids) - set(movie_list))

            negative_item_list = random.sample(
                comp_list, self.negative_sample_count)
            self.negative_sample[str(key)] = negative_item_list
    
    def split_user(self):
        user = np.array(self.users_data[0].unique())
        
        np.random.seed(self.config["numpy_seed"])
        np.random.shuffle(user)
        
        total_len = len(user)
        
        train_user = user[:int(total_len * self.split_ratio)]
        valid_user = user[int(total_len * self.split_ratio):]
        
        return train_user, valid_user
        
    def collect_movie_list(self, user_list):
        ratings = self.ratings_data.iloc[:, 0:3]
        
        data_set = {}

        for user_id in user_list:
            user_ratings = ratings[ratings[0] == user_id]   ## collect movie list and ratings for the user id

            user_positive_movies = list(map(lambda x: str(x), user_ratings[1]))
            data_set[user_id] = user_positive_movies
        
        return data_set
    
    def make_mask_seq(self, data_set, phase):
        '''
        Make masked sequence
        x : I1, I2, I3, I4 -> y : I1, I2, <mask>, I4 
        mask : 0, 0, 1, 0
        pad : 1, 0, 0, 0
        '''
        masking_prob = 0.2
        sample_ratio = 10
        
        sequence_length = self.config["sequence_length"]
        
        x_array = np.empty((0, sequence_length), dtype = str)
        y_array = np.empty((0, sequence_length), dtype = str)
        mask_array = np.empty((0, sequence_length), dtype = int)
        pad_array = np.empty((0, sequence_length), dtype = int)
        
        keys = data_set.keys()
        
        for key in tqdm(keys):
            movie_list = data_set[key]
            movie_length = len(movie_list)

            if movie_length >= 2:
                sample_count = (movie_length) // sample_ratio
                if sample_count == 0:
                    sample_count = 1

                '''
                Select pivot index to slice inputs
                pivot_idx : last index of sequence input
                '''
                pivot_idx_list = []
                pivot_idx_list = random.sample(
                                range(1, movie_length), sample_count)

                for pivot_idx in pivot_idx_list:
                    first_input_idx = pivot_idx + 1 - sequence_length
                    
                    x = []
                    y = []
                    
                    '''
                    Make Paddings
                    '''
                    # Make item vectors 
                    padding_length = 0
                    for idx in range(first_input_idx, pivot_idx+1):
                        if idx < 0:
                            padding_length += 1
                            x.append("pad")
                            y.append("pad")
                        else:
                            x.append(movie_list[idx])
                            y.append(movie_list[idx])
                    pad = np.zeros((1, sequence_length), dtype = int)
                    for i in range(padding_length):
                        pad[0][i] = 1

                            
                    '''
                    Make Masks
                    - While training, masks are randomly choosen.
                    - In validation step, mask is placed at the last item.
                    '''
                    mask = np.zeros((1, sequence_length), dtype = int)
                    
                    if phase == "train":
                        masking_count = int((sequence_length - padding_length) * masking_prob)
                        if masking_count == 0:
                            masking_count = 1
                            
                        masking_idx_list = random.sample(
                                            range(0, sequence_length), masking_count)
                        for mask_idx in masking_idx_list:
                            mask[0][mask_idx] = 1
                            x[mask_idx] = "mask"
                    else:
                        mask[0][sequence_length - 1] = 1
                        x[sequence_length - 1] = "mask"
                    
                    
                    # Reshape and append
                    x = np.reshape(x, (1, -1))
                    y = np.reshape(y, (1, -1))

                    x_array = np.append(x_array, x, axis = 0)
                    y_array = np.append(y_array, y, axis = 0)
                    mask_array = np.append(mask_array, mask, axis = 0)
                    pad_array = np.append(pad_array, pad, axis = 0)

        return x_array, y_array, mask_array, pad_array

    
    def make_seq_to_seq(self, data_set, ratio):
        '''
        Make sequence to sequence inputs
        I1, I2, I3 -> I2, I3, I4
        or
        <pad> I1, I2 -> <pad> I2, I3
        '''
        sequence_length = self.config["sequence_length"]
        
        keys = data_set.keys()
        
        x_array = np.empty((0, sequence_length), dtype=str)
        y_array = np.empty((0, sequence_length), dtype=str)
        u_array = []
        
        for key in tqdm(keys):
            movie_list = data_set[key]
            movie_length = len(movie_list)

            if movie_length >= 2:
                sample_count = (movie_length) // ratio
                if sample_count == 0:
                    sample_count = 1

                pivot_idx_list = []
                pivot_idx_list = random.sample(
                        range(1, movie_length), sample_count)

                for pivot_idx in pivot_idx_list:
                    first_input_idx = pivot_idx - sequence_length
                    
                    x = []
                    y = []
                    for idx in range(first_input_idx, pivot_idx):
                        if idx < 0:
                            x.append("pad")
                            y.append("pad")
                        else:
                            x.append(movie_list[idx])
                            y.append(movie_list[idx+1])

                    x = np.reshape(x, (1, -1))
                    y = np.reshape(y, (1, -1))

                    x_array = np.append(x_array, x, axis=0)
                    y_array = np.append(y_array, y, axis=0)
                    u_array.append(str(key))

        return x_array, y_array, u_array
        
        
    def make_seq_to_one(self, data_set, sampling, ratio=2):
        '''
        Make sequence to one label dataset
        
        EX) Input Sequence = 4
        
        I1, I2, I3, I4 -> I5
        
        Repeat input sampling in ratio of session length
        '''
        sequence_length = self.config["sequence_length"]

        keys = data_set.keys()

        x_array = np.empty((0, sequence_length), dtype=int)
        y_array = np.empty((0, 1), dtype=int)

        for key in keys:
            movie_list = data_set[key]
            movie_length = len(movie_list)

            if movie_length >= sequence_length + 1:
                sampling_ratio = ratio
                sample_count = (
                    movie_length - sequence_length) // sampling_ratio
                if sample_count == 0:
                    sample_count = 1

                output_idx_list = []
                if sampling:
                    output_idx_list = random.sample(
                        range(sequence_length, movie_length), sample_count)
                else:
                    output_idx_list = range(sequence_length, movie_length)

                for output_idx in output_idx_list:
                    input_idx = output_idx - sequence_length

                    x = np.array(movie_list[input_idx: output_idx])
                    y = np.array(movie_list[output_idx])
                    x = np.reshape(x, (1, -1))
                    y = np.reshape(y, (1, -1))

                    x_array = np.append(x_array, x, axis=0)
                    y_array = np.append(y_array, y, axis=0)

        '''
        Output Shape : 
            x : (Total, SEQ) // y : (Total, 1)
        '''
        return x_array, y_array
        
        
    def one_hot_encoding(self, id_list):
        ## 1 in ground truth index, 0 in others
        one_hot_matrix = np.empty((0, self.movie_length+1), dtype = np.int32)
        
        for id in id_list:
            one_hot_vector = np.zeros((1, self.movie_length+1), dtype=np.int32)
            one_hot_vector[0, id] = 1
            
            one_hot_matrix = np.append(one_hot_matrix, one_hot_vector, axis=0)
        
        return one_hot_matrix

    
    def get_dataset(self, phase):
        
        batch_size = self.batch_size
        
        if phase == "train":
            x = self.string_lookup.str_to_idx(self.train_x)
            y = self.string_lookup.str_to_idx(self.train_y)
            
            dataset = tf.data.Dataset.from_tensor_slices((x, y, self.train_m, self.train_p))
            dataset = dataset\
                        .batch(batch_size, drop_remainder=True)\
                        .shuffle(buffer_size = len(x))\
                        .cache()\
                        .prefetch(tf.data.AUTOTUNE)
            
            return dataset
        
        elif phase == "valid":
            x = self.string_lookup.str_to_idx(self.valid_x)
            y = self.string_lookup.str_to_idx(self.valid_y)

            dataset = tf.data.Dataset.from_tensor_slices((x, y, self.valid_m, self.valid_p))
            dataset = dataset\
                        .batch(batch_size, drop_remainder=True)\
                        .cache()\
                        .prefetch(tf.data.AUTOTUNE)
            
            return dataset

    '''
    APIs
    '''
    def get_movie(self, id):
        idx = self.movies_data.index[self.movies_data[0] == id]
        idx = idx.tolist()[0]
        movie = self.movies_data.iat[idx, 1]

        return movie

    def get_movie_len(self):
        return self.movie_len

    def get_movie_ids(self):        
        return self.movie_ids
    
    def get_string_lookup(self):
        return self.string_lookup

    def get_negative_sample(self, user_batch):
        user_list = user_batch.numpy().tolist()
        user_string = list(map(lambda x: x.decode('UTF-8'), user_list))
        
        samples = tf.zeros([0, self.negative_sample_count], dtype = tf.int32)
        
        for user in user_string:
            negative_sample_str = self.negative_sample[user] 
            negative_sample_idx = self.string_lookup.str_to_idx(negative_sample_str)
            negative_sample_idx = tf.cast(negative_sample_idx, dtype = tf.int32)
            negative_sample_idx = tf.reshape(negative_sample_idx, [1,-1])

            samples = tf.concat([samples, negative_sample_idx], axis = 0)

        return samples
        
    
    
    def get_user_movie(self, user_id):
        ratings = self.ratings_data.iloc[:, 0:3]
                
        user_ratings = ratings[ratings[0] == user_id]   ## collect movie list and ratings for the user id
        
        user_positive_movies = np.array(user_ratings[1])
        user_ratings = np.array(user_ratings[2])
        
        return user_positive_movies, user_ratings


    def get_movie_length(self):
        return self.movie_length
    
    def get_user_length(self):
        return self.user_length