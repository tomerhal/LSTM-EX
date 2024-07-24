import torch


class BaseLocalInference(torch.nn.Module):
    def __init__(self,embedding_size):
        super().__init__()

        self.energy_metrix = None


        self.zero_dim_softmax = torch.nn.Softmax(0)
        self.first_dim_softmax = torch.nn.Softmax(1)



    def create_wave_vectors(self,a_vector,b_vector,energy_metrix):
        energized_b_vector = self.first_dim_softmax(energy_metrix) 
        vector_wave_a = torch.matmul(energized_b_vector, b_vector)

        energized_a_vector = self.zero_dim_softmax(energy_metrix).T   # we transform here because b is on dim 1 of the energy mat
        vector_wave_b = torch.matmul(energized_a_vector, a_vector)
        return vector_wave_a,vector_wave_b
    
    def create_m_vector(self,vector_hat,vector_wave):
        sub_vec = vector_hat-vector_wave
        mul_vec = vector_hat*vector_wave
        m_vector = torch.concat((vector_hat,vector_wave,sub_vec,mul_vec),dim=0)     #if we use batchs dim = 1
        return m_vector

    def forward(self, a_hat: torch.Tensor, b_hat: torch.Tensor):
        self.energy_metrix = torch.matmul(a_hat,b_hat.T)
        vector_wave_a,vector_wave_b = self.create_wave_vectors(a_hat,b_hat,self.energy_metrix)
        m_a = self.create_m_vector(a_hat,vector_wave_a)     #TODO gets h_t-1 l/r for tree
        m_b = self.create_m_vector(b_hat,vector_wave_b)
        return m_a, m_b