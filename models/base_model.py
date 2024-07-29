import torch
from inference_levels.InputEncoders.input_encoding import BaseInputEncoder
from inference_levels.local_inferencers.local_inference import BaseLocalInference
from inference_levels.inference_compositors.base_inference_composition import (
    BaseInferenceComposition,
)
from inference_levels.predictors.predictor import BasePredictor


class BaseModel(torch.nn.Module):
    def __init__(
        self,
        input_encoding: BaseInputEncoder,
        local_inference: BaseLocalInference,
        inference_composition: BaseInferenceComposition,
        predictor: BasePredictor,
    ):
        super().__init__()
        self.input_encoding = input_encoding
        self.local_inference = local_inference
        self.inference_composition = inference_composition
        self.predictor = predictor

    """@abc.abstractmethod
    def abst(self, x: torch.Tensor):
        raise NotImplementedError"""

    def forward(self, premise: torch.Tensor, hypothesis: torch.Tensor):
        a_hat, b_hat = self.input_encoding(premise, hypothesis)
        m_a, m_b = self.local_inference(a_hat, b_hat)
        v_a, v_b = self.inference_composition(m_a, m_b)  # TODO gets h_t-1 l/r for tree
        prediction = self.predictor(v_a, v_b)
        return prediction
