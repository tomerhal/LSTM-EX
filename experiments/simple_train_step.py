from itertools import chain
import torch


def simple_train_step(optimizer,criterion,batch_size,SEQ_DIM,device,model):
        
    def train_step(x, y):
        first_hidden = torch.rand((2,batch_size,50)).to(device)
        decoder_input = torch.rand((batch_size,2)).to(device)
        model.to(device)

        y_hat, attention_output, attention_outputs = model(x,first_hidden,decoder_input)
        
        loss = criterion(y_hat, torch.nn.functional.one_hot(y.reshape(-1,1).long(),num_classes=2).squeeze(1).float())
        
        
        loss.backward()
        
        
        optimizer.step()
        optimizer.zero_grad()
    
    
        return loss.item(), attention_output, attention_outputs

    return train_step
