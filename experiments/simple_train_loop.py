from . import simple_train_step as train_step

import matplotlib.pyplot as plt
import numpy as np

def simple_train_loop(train_loader, optimizer,criterion,device,model,split_val,num_of_epochs,is_test=False,metrices={}):

    losses = []
    losses_plt = []
    train_step_fuc = train_step.simple_train_step(optimizer,criterion,device,model,split_val,is_test,metrices)
    for epoch in range(num_of_epochs):
        losses = []
        for x_batch, y_batch in train_loader:

            x_batch = x_batch.to(device)
            y_batch = y_batch.to(device)
            
            loss = train_step_fuc(x_batch, y_batch)
            #print("lossssssssssssssssss",loss)
            losses.append(loss)

        losses_plt.append(np.mean(losses))
        print("mean",np.mean(losses))
    plt.plot(losses_plt)
    plt.show()
    
        
    
