import torch


class BaseInputEncoder(torch.nn.Module):
    def __init__(self, encoding_layer):
        super().__init__()

        self.encoding_layer = encoding_layer

    def forward(self, a_vector: torch.Tensor, b_vector: torch.Tensor):
        a_hat = self.encoding_layer(a_vector)
        b_hat = self.encoding_layer(b_vector)
        return a_hat, b_hat
