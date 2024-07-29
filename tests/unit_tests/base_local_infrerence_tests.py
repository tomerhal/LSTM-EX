
import torch


def test_create_wave_vectors(bli,a_hat,b_hat,energy_metrix):

    a_wave,b_wave = bli.create_wave_vectors(a_hat,b_hat,energy_metrix)
    print(a_wave,b_wave)

def test_create_m_vector():
    print()

def test_forward():
    print()

def test_energy_metrix():
    print()

def test_all():
    print()


def main(bli,hidden_size):

    a_seq_len = 2
    b_seq_len = 3
    


    a_hat = torch.tensor([[1.0,2.0],[3.0,4.0]])
    b_hat = torch.tensor([[1.0,2.0],[3.0,4.0],[4.0,5.0]])
    energy_metrix = torch.matmul(a_hat,b_hat.T)
    test_create_wave_vectors(bli,a_hat,b_hat,energy_metrix)