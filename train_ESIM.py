import experiments.simple_train_exp as train_exp
from inference_levels.InputEncoders.ESIM_encoder import ESIMInputEncoder
from inference_levels.local_inferencers.local_inference import BaseLocalInference
from inference_levels.inference_compositors.base_inference_composition import BaseInferenceComposition
from inference_levels.predictors.predictor import BasePredictor
from data.handle_data import get_datasets
#from experiments.simple_train_exp import run_simple_train_exp
from models.ESIM_model import ESIMModel
import torch
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import numpy as np

#train_data = torch.rand((2,3,3))
#train_loader = DataLoader(dataset=[train_data[0],train_data[1]], batch_size=1, shuffle=True)
train_data, test_data = get_datasets()
train_loader = DataLoader(dataset=train_data, batch_size=1, shuffle=True)

input_size,hidden_size = 3,3
input_encoding = ESIMInputEncoder(input_size,hidden_size)
local_inference = BaseLocalInference(hidden_size)
composition_layer = torch.nn.LSTM(input_size, hidden_size, num_layers=1, batch_first=False, dropout=0.0, bidirectional=True)
infrence_composition = BaseInferenceComposition(hidden_size,composition_layer)
predictor = BasePredictor()
model = ESIMModel(input_encoding,local_inference,infrence_composition,predictor)
train_exp.run_simple_train_exp(train_loader,model)
