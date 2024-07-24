import torch
from .input_encoding import BaseInputEncoder

class ESIMInputEncoder(BaseInputEncoder):
    def __init__(self,input_size,hidden_size):
        encoding_layer = torch.nn.LSTM(input_size, hidden_size, num_layers=1, batch_first=False, dropout=0.0, bidirectional=True)
        
        super().__init__(encoding_layer)



    def forward(self, a_vector: torch.Tensor, b_vector: torch.Tensor, h0, c0):
        a_hat, hidden_a =  self.encoding_layer(a_vector, h0, c0)
        b_hat, hidden_b =  self.encoding_layer(b_vector, h0, c0)
        return a_hat, b_hat
    

        #hidden = torch.cat((hidden[-2,::], hidden[-1,::]), dim = 1)