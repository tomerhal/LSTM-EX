import torch
from .input_encoding import BaseInputEncoder

class ESIMInputEncoder(BaseInputEncoder):
    def __init__(self,input_size,hidden_size):
        encoding_layer = torch.nn.LSTM(input_size, hidden_size, num_layers=1, batch_first=True, dropout=0.0, bidirectional=True)
        super().__init__(encoding_layer)



    def forward(self, a_vector: torch.Tensor, b_vector: torch.Tensor, h0_p, c0_p, h0_h, h1_h):
        a_hat, hidden_a =  self.encoding_layer(a_vector)
        b_hat, hidden_b =  self.encoding_layer(b_vector)

        return a_hat, b_hat
    