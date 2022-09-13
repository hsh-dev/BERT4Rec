from DataModule.DataLoader import DataLoader
from TrainModule.TrainManager import TrainManager
from config import config
from Models.BERT_Model import BERT

if __name__ == "__main__":

    dataloader = DataLoader(config)
    item_count = dataloader.get_movie_len()
        
    model = BERT(i_dim = item_count,
                 n_dim = config["sequence_length"],
                 d_dim = config["hidden_dim"],
                 h_num = 4,
                 l_num = 2)    
    
    trainmanger = TrainManager(model, dataloader, config)

    best_score = trainmanger.start()
    