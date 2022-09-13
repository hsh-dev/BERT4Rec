from DataModule.DataLoader import DataLoader
from TrainModule.TrainManager import TrainManager
# from Models.SAS_Model import SAS
from config import config
from Models.BERT_Model import BERT

if __name__ == "__main__":

    dataloader = DataLoader(config)
    item_count = dataloader.get_movie_len()
        
        
    model = BERT(i_dim = item_count,
                 n_dim = config["sequence_length"],
                 d_dim = config["hidden_dim"],
                 h_num = 4,
                 l_num = 4)    
    
    # model = SAS(i_dim = item_count, 
    #             d_dim = config["hidden_dim"], 
    #             n_dim = config["sequence_length"], 
    #             b_num = config["attention_layer_count"])
    
    trainmanger = TrainManager(model, dataloader, config)

    best_score = trainmanger.start()
    