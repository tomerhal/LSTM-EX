import torch


class BaseInputEncoder(torch.nn.Module):
    def __init__(self,embedding_size, encoding_layer):
        super().__init__()

        self.encoding_layer = encoding_layer



    def forward(self, m_a: torch.Tensor, m_b: torch.Tensor):
        a_hat =  self.composition_layer(self.activation_layer(self.FF_layer(m_a)))
        b_hat =  self.composition_layer(self.activation_layer(self.FF_layer(m_b)))
        return a_hat, b_hat