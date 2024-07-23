import torch


class BaseLocalInference(torch.nn.Module):
    def __init__(self,embedding_size, composition_layer, FF_layer = None, activation_layer = None):
        super().__init__()
        if FF_layer != None:
            self.FF_layer = FF_layer
        else:
            self.FF_layer= torch.nn.Linear(4*embedding_size,embedding_size)

        self.energy_metrix = torch.zeros((embedding_size,embedding_size))

        self.composition_layer = composition_layer
        self.zero_dim_softmax = torch.nn.Softmax(0)
        self.first_dim_softmax = torch.nn.Softmax(1)



    def create_wave_vectors(self,a_vector,b_vector):
        energized_b_vector = self.zero_dim_softmax(self.energy_metrix) * b_vector
        vector_wave_a = torch.sum(energized_b_vector,dim=0)

        energized_a_vector = self.first_dim_softmax(self.energy_metrix).T * a_vector    # we transform here because b is on dim 1 of the energy mat
        vector_wave_b = torch.sum(energized_a_vector,dim=0)
        return vector_wave_a,vector_wave_b
    
    def create_m_vector(self,vector_hat,vector_wave):
        sub_vec = vector_hat-vector_wave
        mul_vec = vector_hat*vector_wave
        m_vector = torch.concat((vector_hat,vector_wave,sub_vec,mul_vec),dim=0)     #if we use batchs dim = 1
        return m_vector

    def forward(self, a_hat: torch.Tensor, b_hat: torch.Tensor):
        self.energy_metrix = torch.matmul(a_hat,b_hat)
        vector_wave_a,vector_wave_b = self.create_wave_vectors(a_hat,b_hat)
        m_a = self.create_m_vector(a_hat,vector_wave_a)
        m_b = self.create_m_vector(b_hat,vector_wave_b)
        return m_a, m_b