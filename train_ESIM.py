import experiments.simple_train_exp as train_exp
from inference_levels.InputEncoders.ESIM_encoder import ESIMInputEncoder
from inference_levels.local_inferencers.local_inference import BaseLocalInference
from inference_levels.inference_compositors.base_inference_composition import BaseInferenceComposition
from inference_levels.predictors.predictor import BasePredictor
from data.handle_data import get_datasets,get_datasets_from_pkl
#from experiments.simple_train_exp import run_simple_train_exp
from models.ESIM_model import ESIMModel
import torch
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import numpy as np

BATCH_SIZE = 64
INPUT_SIZE,HIDDEN_SIZE = 300,300
device = 'cuda' if torch.cuda.is_available() else 'cpu'
train_data, test_data, split_val = get_datasets()
train_loader = DataLoader(dataset=train_data, batch_size=BATCH_SIZE, shuffle=True)


input_encoding = ESIMInputEncoder(INPUT_SIZE,HIDDEN_SIZE)
local_inference = BaseLocalInference(HIDDEN_SIZE)
composition_layer = torch.nn.LSTM(INPUT_SIZE, HIDDEN_SIZE, num_layers=1, batch_first=False, dropout=0.0, bidirectional=True)
infrence_composition = BaseInferenceComposition(HIDDEN_SIZE,composition_layer)
predictor = BasePredictor()
model = ESIMModel(input_encoding,local_inference,infrence_composition,predictor)
train_exp.run_simple_train_exp(train_loader,model,split_val)
torch.save(model.state_dict(),f="trained_model.pkl")
