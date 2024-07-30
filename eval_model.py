from experiments.simple_train_loop import simple_train_loop
from itertools import chain
from torch import nn
import torch
from torch.utils.data import Dataset, TensorDataset,DataLoader
import pickle
from data.handle_data import get_datasets,get_datasets_from_pkl
from models.ESIM_model import ESIMModel
from inference_levels.InputEncoders.ESIM_encoder import ESIMInputEncoder
from inference_levels.local_inferencers.local_inference import BaseLocalInference
from inference_levels.inference_compositors.base_inference_composition import BaseInferenceComposition
from inference_levels.predictors.predictor import BasePredictor


device = torch.device( 'cuda' if torch.cuda.is_available() else 'cpu' )
INPUT_SIZE,HIDDEN_SIZE = 300,300
input_encoding = ESIMInputEncoder(INPUT_SIZE,HIDDEN_SIZE)
local_inference = BaseLocalInference(HIDDEN_SIZE)
composition_layer = torch.nn.LSTM(INPUT_SIZE, HIDDEN_SIZE, num_layers=1, batch_first=False, dropout=0.0, bidirectional=True)
infrence_composition = BaseInferenceComposition(HIDDEN_SIZE,composition_layer)
predictor = BasePredictor()
model = ESIMModel(input_encoding,local_inference,infrence_composition,predictor)
model.load_state_dict(torch.load('trained_model.pkl'))


train_data, test_data, split_val = get_datasets_from_pkl()
test_loader = DataLoader(dataset=test_data, batch_size=len(test_data), shuffle=True)

optimizer = torch.optim.Adam(chain(model.parameters()), lr=0.0001)
criterion = nn.CrossEntropyLoss()
num_of_epochs = 1
metrices = {}
metrices['acurrecy'] = lambda y_hat_y_batch: print((torch.argmax(y_hat_y_batch[0],dim=1) == torch.argmax(y_hat_y_batch[1],dim=1)).float().mean())
print(metrices.items())
simple_train_loop(test_loader, optimizer,criterion,device,model,split_val,num_of_epochs,is_test=True,metrices=metrices)