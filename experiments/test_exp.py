import torch
from torch.utils.data import Dataset, TensorDataset,DataLoader
import numpy as np

# evaluate model at end of epoch

def eval_test(test_loader,model,batch_size,device):
    losses = []
    with torch.no_grad():
        for x_batch, y_batch in test_loader:
            # the dataset "lives" in the CPU, so do our mini-batches
            # therefore, we need to send those mini-batches to the
            # device where the model "lives"
            x_batch = x_batch.to(device)
            y_batch = y_batch.to(device)
            
            first_hidden = torch.zeros((2,batch_size,50)).to(device)
            decoder_input = torch.zeros((batch_size,1)).to(device)
            #model.to(device)
            y_hat = model(premise, hypothesis,first_hidden_pre,decoder_input_pre,first_hidden_hyp,decoder_input_hyp)
            #y_hat, items, attention_output = model(x_batch,first_hidden,decoder_input)
            #y_hat= model2(x_batch)
            #y_pred = model(x_batch)
            acc = (torch.argmax(y_hat,dim=1) == y_batch).float().mean()
            acc = float(acc)
            print(f"End of batch, accuracy {acc}")
            print(torch.argmax(y_hat,dim=1),y_hat,y_batch)
            losses.append(acc)
        print(np.mean(acc))
