import abc
import torch

from models.base_model import BaseModel


class ESIMModel(BaseModel):
    def __init__(
        self, input_encoding, local_inference, infrence_composition, predictor
    ):

        super().__init__(
            input_encoding, local_inference, infrence_composition, predictor
        )

    def forward(self, premise: torch.Tensor, hypothesis: torch.Tensor, h0, c0):
        a_hat, b_hat = self.input_encoding(premise, hypothesis, h0, c0)
        m_a, m_b = self.local_inference(a_hat, b_hat)
        v_a, v_b = self.inference_composition(
            m_a, m_b, h0, c0
        )  # TODO gets h_t-1 l/r for tree
        prediction = self.predictor(v_a, v_b)
        return prediction
