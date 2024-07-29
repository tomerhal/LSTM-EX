import abc

import torch


class BaseInferenceComposition(torch.nn.Module):
    def __init__(
        self, embedding_size, composition_layer, FF_layer=None, activation_layer=None
    ):
        super().__init__()
        if FF_layer != None:
            self.FF_layer = FF_layer
        else:
            self.FF_layer = torch.nn.Linear(4 * embedding_size, embedding_size)

        if activation_layer != None:
            self.activation_layer = activation_layer
        else:
            self.activation_layer = torch.nn.ReLU()

        self.composition_layer = composition_layer

    def forward(self, m_a: torch.Tensor, m_b: torch.Tensor):
        v_a = self.composition_layer(
            self.activation_layer(self.FF_layer(m_a))
        )  # TODO gets h_t-1 l/r for tree
        v_b = self.composition_layer(self.activation_layer(self.FF_layer(m_b)))
        return v_a, v_b
