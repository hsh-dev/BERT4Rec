import tensorflow as tf
import numpy as np
import time
import datetime

from TrainModule.Scheduler import CosineDecayWrapper
from TrainModule.LossManager import LossManager
from TrainModule.ScoreManager import ScoreManager
import keras

class TrainManager():
    def __init__(self, model, dataloader, config) -> None:
        self.config = config    
        self.model = model
        self.dataloader = dataloader

        self.batch_size = config["batch_size"]
        self.loss = config["loss"]
        self.embedding = config["embedding"]
        
        self.loss_manager = LossManager()
        self.score_manager = ScoreManager()
        
        self.optimizer_init()
        
        self.log = {}
    
    def optimizer_init(self):
        optimizer = None
        if self.config["optimizer"] == "ADAM":
            optimizer = tf.keras.optimizers.Adam(
                        learning_rate=self.config["learning_rate"], beta_1=0.9, beta_2=0.999)
        elif self.config["optimizer"] == "SGD":
            optimizer = tf.keras.optimizers.SGD(
                        learning_rate=self.config["learning_rate"], momentum=0.9)
        else:
            raise Exception("Write Appropriate Optimizer")

        self.optimizer_wrap = CosineDecayWrapper(
                optimizer= optimizer,
                max_lr=self.config["learning_rate"],
                min_lr=self.config["min_learning_rate"],
                max_epochs=self.config["max_epoch"],
                decay_cycles=self.config["decay_cycle"],
                decay_epochs=self.config["decay_epoch"]
        )
    
    def start(self):
        total_epoch = self.config["max_epoch"]

        min_valid_loss = 9999
        save_valid_hr = None
        
        not_update_count = 0
        
        for epoch in range(total_epoch):
            print("\n# Epoch {} #".format(epoch+1))
            print("## Train Start ##")
            self.train_loop("train")
            print("Train Loss : {} \n".format(self.log["train_loss"]))

            print("## Validation Start ##")
            self.train_loop("valid")
            print("Valid Loss : {} \n".format(self.log["valid_loss"]))
            self.print_hit_rate(self.log["valid_hr"], self.valid_total_user)
            
            self.optimizer_wrap.update_lr(epoch)
            
            if self.log["valid_loss"] < min_valid_loss:
                not_update_count = 0
                save_valid_hr = self.log["valid_hr"]
                min_valid_loss = self.log["valid_loss"]
            else:
                not_update_count += 1
            
            if not_update_count >= 20:
                print("No update on valid loss. Early stop...")
                break
            
        print("[Best Validation Hit Rate]")
        self.print_hit_rate(save_valid_hr, self.valid_total_user)
        
        return save_valid_hr
        
                
    def train_loop(self, phase):
        if phase == "train":
            dataset = self.dataloader.get_dataset("train")
            self.model.trainable = True
        else:
            dataset = self.dataloader.get_dataset("valid")
            
        total_step = len(dataset)
        if phase == "valid":
            self.valid_total_user = total_step * self.batch_size
            
        print_step = total_step // 3
        
        all_loss_list = []
        loss_list = []
        
        hr_keys = ['100', '50', '20', '10', '5', '3']
        
        all_hr_dict = {}
        
        init_hr_dict = {}
        for key in hr_keys:
            all_hr_dict[key] = 0
            init_hr_dict[key] = 0
        hr_dict = init_hr_dict
        
        start_time = time.time()

        for idx, sample in enumerate(dataset):
            x, y, mask, pad = sample       

            loss, y_pred, hr_number = self.propagate_with_graph(x, y, mask, pad, phase, k = 10)
                
            all_loss_list.append(loss)
            loss_list.append(loss)
            
            if phase == "valid":
                for key in hr_keys:
                    all_hr_dict[key] += hr_number[key]
                    hr_dict[key] += hr_number[key]

            if (idx+1) % print_step == 0:
                end_time = time.time()
                
                losses = np.average(np.array(loss_list))
                print("STEP: {}/{} | Loss: {} | Time: {}s".format(
                                                                idx+1, 
                                                                total_step, 
                                                                round(losses, 5), 
                                                                round(end_time - start_time, 5)
                                                                ))
                if phase == "valid":
                    self.print_hit_rate(hr_dict, print_step * self.batch_size)
                    hr_dict = init_hr_dict
                    
                loss_list.clear()
                start_time = time.time()
                    
        total_loss = np.average(np.array(all_loss_list))
        
        if phase == "valid":
            self.log["valid_hr"] = all_hr_dict
        self.save_logs(total_loss, phase)
    
    def print_hit_rate(self, hr_dict, total_user):
        print("HR@3 : {} | HR@5 : {} | HR@10 : {} |".format(
                                                        round(hr_dict['3'].numpy()/total_user, 5),
                                                        round(hr_dict['5'].numpy()/total_user, 5),
                                                        round(hr_dict['10'].numpy()/total_user, 5)))
        print("HR@20 : {} | HR@50 : {} | HR@100 : {} |\n".format(
                                                        round(hr_dict['20'].numpy()/total_user, 5),
                                                        round(hr_dict['50'].numpy()/total_user, 5),
                                                        round(hr_dict['100'].numpy()/total_user, 5)))


    def make_one_hot_vector(self, y, dim):
        dim = tf.cast(dim, dtype = tf.int32)        
        one_hot = tf.one_hot(y, dim)
        return one_hot

    @tf.function
    def propagate_with_graph(self, x, y, mask, pad, phase, k):
        loss, y_pred = self.propagation(x, y, mask, pad, phase)
        
        hit_rate = None
        if phase == "valid":
            hit_rate = self.score_manager.hit_rate(y, y_pred, k)
        
        return loss, y_pred, hit_rate


    def propagation(self, x, y_true, mask, pad, phase):
        with tf.GradientTape() as tape:
            y_pred = self.model(x, pad)
            
            # loss = self.loss_manager.bpr_loss(y_true, y_pred)
            loss = self.loss_manager.negative_log_with_mask(y_true, y_pred, mask)
            
        gradients = tape.gradient(loss, self.model.trainable_variables)
        
        if phase == "train":
            self.optimizer_wrap.optimizer.apply_gradients(
                zip(gradients, self.model.trainable_variables))

        return loss, y_pred
    
    '''
    Save Functions
    '''
    def save_logs(self, loss, phase):
        loss_key = phase + "_loss"
        
        self.log[loss_key] = loss
    