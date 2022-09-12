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
        save_valid_hr = 0
        
        not_update_count = 0
        
        for epoch in range(total_epoch):
            print("\n# Epoch {} #".format(epoch+1))
            print("## Train Start ##")
            self.train_loop("train")
            print("Train Loss : {} | HR@10 : {} \n".format(self.log["train_loss"], self.log["train_hr"]))

            print("## Validation Start ##")
            self.train_loop("valid")
            print("Valid Loss : {} | HR@10 : {}".format(self.log["valid_loss"], self.log["valid_hr"]))
            
            self.optimizer_wrap.update_lr(epoch)
            
            if self.log["valid_loss"] < min_valid_loss:
                not_update_count = 0
                save_valid_hr = self.log["valid_hr"]
            else:
                not_update_count += 1
            
            print("Best Validation Hit Rate : {}".format(save_valid_hr))
             
            if not_update_count >= 20:
                print("No update on valid loss. Early stop...")
                break
        return save_valid_hr
        
                
    def train_loop(self, phase):
        if phase == "train":
            dataset = self.dataloader.get_dataset("train")
            self.model.trainable = True
        else:
            dataset = self.dataloader.get_dataset("valid")
            
        total_step = len(dataset)
        print_step = total_step // 3
        
        all_loss_list = []
        loss_list = []
        all_hr_list = []
        hr_list = []
        
        start_time = time.time()

        for idx, sample in enumerate(dataset):
            x, y = sample
            # n_s = self.dataloader.get_negative_sample(u)  
            
            # if idx == 0:
            #     ## Trace Initialize
            #     stamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
            #     logdir = 'log/func/%s' % stamp
            #     writer = tf.summary.create_file_writer(logdir)
            #     tf.summary.trace_on(graph=True, profiler=True)
            
            loss, y_pred, hr = self.propagate_with_graph(x, y, phase, k = 10)
            # if idx == 0:
            #     ## Write
            #     with writer.as_default():
            #         tf.summary.trace_export(name="my_func_trace",
            #                                 step=0,
            #                                 profiler_outdir=logdir)
            #     tf.summary.trace_off()
                
            all_loss_list.append(loss)
            loss_list.append(loss)
            all_hr_list.append(hr)
            hr_list.append(hr)
            
            if (idx+1) % print_step == 0:
                end_time = time.time()
                
                losses = np.average(np.array(loss_list))
                hr_avg = np.average(np.array(hr_list))
                print("STEP: {}/{} | Loss: {} | Time: {}s".format(
                                                                idx+1, 
                                                                total_step, 
                                                                round(losses, 7), 
                                                                round(end_time - start_time, 5)
                                                                ))
                print("HR: {} ".format(round(hr_avg, 7)))
                
                loss_list.clear()
                hr_list.clear()
                start_time = time.time()
                    
        total_loss = np.average(np.array(all_loss_list))
        total_hr = np.average(np.array(all_hr_list))
        
        self.save_logs(total_loss, total_hr, phase)
    

    def make_one_hot_vector(self, y, dim):
        dim = tf.cast(dim, dtype = tf.int32)        
        one_hot = tf.one_hot(y, dim)
        return one_hot

    @tf.function
    def propagate_with_graph(self, x, y, phase, k):
        loss, y_pred = self.propagation(x, y, phase)
        
        hit_rate = self.score_manager.hit_rate(y, y_pred, k)
        
        return loss, y_pred, hit_rate


    def propagation(self, x, y_true, phase):
        with tf.GradientTape() as tape:
            y_pred = self.model(x)
                        
            loss = self.loss_manager.bpr_loss(y_true, y_pred)
 
        gradients = tape.gradient(loss, self.model.trainable_variables)
        
        if phase == "train":
            self.optimizer_wrap.optimizer.apply_gradients(
                zip(gradients, self.model.trainable_variables))

        return loss, y_pred
    
    '''
    Save Functions
    '''
    def save_logs(self, loss, hr, phase):
        loss_key = phase + "_loss"
        hr_key = phase + "_hr"
        
        self.log[loss_key] = loss
        self.log[hr_key] = hr
    