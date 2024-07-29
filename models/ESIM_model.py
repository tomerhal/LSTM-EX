import abc
import torch

from models.base_model import BaseModel

class ESIMModel(BaseModel):
    def __init__(self, input_encoding,local_inference,infrence_composition,predictor):
        
        super().__init__(input_encoding,local_inference,infrence_composition,predictor)


    def forward(self, premise: torch.Tensor, hypothesis: torch.Tensor, h0_p, c0_p, h0_h, h1_h):
        a_hat, b_hat = self.input_encoding(premise,hypothesis, h0_p, c0_p, h0_h, h1_h)
        #print('a_hat, b_hat',a_hat, b_hat)
        m_a, m_b = self.local_inference(a_hat,b_hat)
        #print('m_a, m_b',m_a, m_b)
        v_a, v_b = self.inference_composition(m_a,m_b, h0_p, c0_p, h0_h, h1_h)   #TODO gets h_t-1 l/r for tree
        #print('v_a, v_b',v_a, v_b)
        prediction = self.predictor(v_a, v_b)
        return prediction


