import abc

import torch


class Base(torch.nn.Module, abc.ABC):
    def __init__(self, input_encoding: InputEncoding, local_inference: LocalInference, infrence_composition: InferenceComposition, predictor: Predictor):
        super().__init__()
        self.input_encoding
        self.local_inference
        self.inference_composition
        self.predictor
        self.module = torch.nn.Sequential(
            torch.nn.LazyLinear(out_features * 2),
            torch.nn.GELU(),
            torch.nn.LazyLinear(out_features),
        )

    @abc.abstractmethod
    def abst(self, x: torch.Tensor):
        raise NotImplementedError


    def forward(self, premise: torch.Tensor, hypothesis: torch.Tensor):
        a_hat, b_hat = self.input_encoding(premise,hypothesis)
        m_a, m_b = self.local_inference(a_hat,b_hat)
        v_a, v_b = self.inference_composition(m_a,m_b)   #TODO gets h_t-1 l/r for tree
        prediction = self.predictor(v_a, v_b)
        return prediction



class ResnetLike(Base):
    def forward(self, x: torch.Tensor):
        # super().forward(x) would raise an error correctly
        return self.module(x) + x


class TestNetwork(Base):
    def forward(self, x: torch.Tensor):
        return self.module(x) * 2