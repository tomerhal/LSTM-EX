from . import simple_train_loop as train_loop
from itertools import chain
from torch import nn
import torch

def run_simple_train_exp(train_loader,model):
    device = torch.device( 'cuda' if torch.cuda.is_available() else 'cpu' )
    optimizer = torch.optim.Adam(chain(model.parameters()), lr=0.001)
    criterion = nn.BCEWithLogitsLoss()
    batch_size = 1
    SEQ_DIM = 3
    train_loop.simple_train_loop(train_loader,optimizer,criterion,batch_size,SEQ_DIM,device,model)
