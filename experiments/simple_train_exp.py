from . import simple_train_loop as train_loop
from itertools import chain
from torch import nn
import torch

def run_simple_train_exp(train_loader,model,split_val):
    device = torch.device( 'cuda' if torch.cuda.is_available() else 'cpu' )
    optimizer = torch.optim.Adam(chain(model.parameters()), lr=0.0001)
    criterion = nn.CrossEntropyLoss()
    num_of_epochs = 150
    train_loop.simple_train_loop(train_loader,optimizer,criterion,device,model,split_val,num_of_epochs)

