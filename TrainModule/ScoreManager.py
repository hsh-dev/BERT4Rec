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

        metric_k = ['100', '50', '20', '10', '5', '3']
        
        for k in metric_k:
            top_k = int(k)
            y_pred_top = y_pred_sort[:,:top_k]
            y_exist = tf.equal(y_pred_top, y_true)
            y_exist = tf.cast(y_exist, tf.int32)
            hit_number = tf.math.count_nonzero(y_exist)
            hit_number_dict[k] = hit_number
        
        return hit_number_dict