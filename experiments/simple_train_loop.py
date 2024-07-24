from . import simple_train_step as train_step

import numpy as np

def simple_train_loop(train_loader, optimizer,criterion,batch_size,SEQ_DIM,device,model):
#train_loader = DataLoader(dataset=train_data, batch_size=100, shuffle=True)
#train_loader = DataLoader(dataset=[train_data[0],train_data[1]], batch_size=2, shuffle=True)

    losses = []
    att = []
    att_all = []

    train_step_fuc = train_step.simple_train_step(optimizer,criterion,batch_size,SEQ_DIM,device,model)

    for epoch in range(100): #40
        losses = []
        for x_batch, y_batch in train_loader:
            #print(all([all(x) for x in torch.isfinite(x_batch)]))
            x_batch = x_batch.to(device)
            y_batch = y_batch.to(device)
            
            loss, attention_output, attention_outputs = train_step_fuc(x_batch, y_batch)
            print("lossssssssssssssssss",loss)
            losses.append(loss)
            att.append(attention_output)
            att_all.append(attention_output)
        print(np.mean(losses))
        
    
