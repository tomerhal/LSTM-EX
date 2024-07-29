import torch


class BasePredictor(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def get_avg_max_vector(self, vector):
        avg_vec = torch.sum(vector) / len(vector)
        max_vec = torch.max(vector)
        return avg_vec, max_vec

    def forward(self, v_a_vector: torch.Tensor, v_b_vector: torch.Tensor):
        avg_vec_a, max_vec_a = self.get_avg_max_vector(v_a_vector)
        avg_vec_b, max_vec_b = self.get_avg_max_vector(v_b_vector)
        return torch.concat(
            (avg_vec_a, max_vec_a, avg_vec_b, max_vec_b), dim=0
        )  # if we use batchs dim = 1
