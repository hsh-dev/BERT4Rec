import tensorflow as tf


'''
Caculating Accuracy
'''
class ScoreManager():
    def __init__(self) -> None:
        pass
    
    @tf.function
    def hit_rate(self, y_true, y_pred, k, sequence = True):
        '''
        Recording hit if target is in top-k items
        k : number to choose top # items
        
        W/O Sequence
        y_true (Batch,): label of output - index type    
        y_pred (Batch,Items): prediction output
        
        W Sequence
        y_true (Batch, Seq)
        y_pred (Batch, Seq, Items)
        
        Validation Set
        y_pred (Batch, Items)
        '''
        hit_number_dict = {}
        
        y_true = y_true[:,-1]
        y_true = tf.cast(y_true, tf.int32)
        y_true = tf.reshape(y_true, [-1, 1])
        
        y_pred = y_pred[:,-1,:]
        y_pred = tf.squeeze(y_pred)
        y_pred_sort = tf.argsort(y_pred, axis=1, direction='DESCENDING')
        
        total_user = y_pred.shape[0]

        # HR@100
        y_pred_top = y_pred_sort[:,:100]
        
        y_exist = tf.equal(y_pred_top, y_true)
        y_exist = tf.cast(y_exist, tf.int32)
        hit_number = tf.math.count_nonzero(y_exist)
        hit_number_dict['100'] = hit_number
        
        # HR@50
        y_pred_top = y_pred_top[:, :50]
        
        y_exist = tf.equal(y_pred_top, y_true)
        y_exist = tf.cast(y_exist, tf.int32)
        hit_number = tf.math.count_nonzero(y_exist)
        hit_number_dict['50'] = hit_number 
        
        # HR@20
        y_pred_top = y_pred_top[:, :20]

        y_exist = tf.equal(y_pred_top, y_true)
        y_exist = tf.cast(y_exist, tf.int32)
        hit_number = tf.math.count_nonzero(y_exist)
        hit_number_dict['20'] = hit_number 
        
        # HR@10
        y_pred_top = y_pred_top[:, :10]

        y_exist = tf.equal(y_pred_top, y_true)
        y_exist = tf.cast(y_exist, tf.int32)
        hit_number = tf.math.count_nonzero(y_exist)
        hit_number_dict['10'] = hit_number 

        # HR@5
        y_pred_top = y_pred_top[:, :5]

        y_exist = tf.equal(y_pred_top, y_true)
        y_exist = tf.cast(y_exist, tf.int32)
        hit_number = tf.math.count_nonzero(y_exist)
        hit_number_dict['5'] = hit_number 
    
        # HR@3
        y_pred_top = y_pred_top[:, :3]

        y_exist = tf.equal(y_pred_top, y_true)
        y_exist = tf.cast(y_exist, tf.int32)
        hit_number = tf.math.count_nonzero(y_exist)
        hit_number_dict['3'] = hit_number 
        
        return hit_number_dict

        
        