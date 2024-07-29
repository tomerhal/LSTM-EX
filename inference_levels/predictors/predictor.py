import torch


class BasePredictor(torch.nn.Module):
    def __init__(self,FF_layer = None,activation_layer = None):
        super().__init__()
        NUM_OF_CLASSES = 3
        if FF_layer != None:
            self.FF_layer = FF_layer
        else:
            self.FF_layer= torch.nn.Linear(4,NUM_OF_CLASSES)
        torch.nn.init.xavier_uniform(self.FF_layer.weight)
            

        if activation_layer != None:
            self.activation_layer = activation_layer
        else:
            self.activation_layer= torch.nn.Tanh()


    def get_avg_max_vector(self,vector):
        avg_vec = torch.sum(vector,dim=(1,2))/len(vector[0])
        max_vec = torch.max(torch.max(vector,dim=2).values,dim=1,keepdim=False).values
        return avg_vec, max_vec

    def forward(self, v_a_vector: torch.Tensor, v_b_vector: torch.Tensor):
        avg_vec_a, max_vec_a =  self.get_avg_max_vector(v_a_vector)
        avg_vec_b, max_vec_b =  self.get_avg_max_vector(v_b_vector)

        v = torch.stack((avg_vec_a,max_vec_a,avg_vec_b,max_vec_b),dim=1) 

        return self.activation_layer(self.FF_layer(v)).softmax(dim=1)