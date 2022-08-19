# train functions
import numpy as np
import time
import torch
from torch.optim.lr_scheduler import ExponentialLR
from tqdm import tqdm
import wandb

from src.snam.utils import *
from src.snam.snam import *

# define device to train on
#device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def fit_model(trainloader, testloader, n_features, model, optimizer, device, max_epoch, run_id,
        torch_seed=0, test=False, lbd=0.1, MSE=False, save_dir = "../../models/snam/",
        adaptive_lr=False):
    """
    Train and evaluate model on provided train and val sets.
    Input:
    Output:
    """  
    
    wandb.log({"max_epochs" : max_epoch})
    start = time.time()
    
    print("################################## START OF TRAINING ##################################")
    
    n_layer=0
    for name, param in model.named_parameters():
        if 'feature_nns.0' in name and 'weight' in name:
            n_layer+=1
    
    model.train()
    
    if MSE==True:
      criterion=torch.nn.MSELoss()
    else:
      criterion=torch.nn.CrossEntropyLoss()
    
    # save model and loss metrics
    epoch_train_loss_history = []
    epoch_test_loss_history = []
    
    if adaptive_lr:
        scheduler = ExponentialLR(optimizer, gamma=0.9)
    
    # TO-DO: implement early stopping!!
    #early_stopping = EarlyStopping(tolerance=2, min_delta=0)
    
    for epoch in range(max_epoch):
        if (epoch)%5==0:
            print(f"##### Training Epoch Nr: {epoch+1} started ####")
            #print("Early stopping counter: ", early_stopping.counter)
        f_out_tr = []                 # train feature outputs
        total_batch_loss=0
        tr_correct, tr_total = 0, 0
        
        for idx1, (inputs, targets) in tqdm(enumerate(trainloader)): 
            inputs, targets = inputs.to(device), targets.to(device)
            optimizer.zero_grad()
            outputs, f_out_tr_temp = model(inputs)[0],model(inputs)[1].detach().cpu().numpy()
            f_out_tr.append(f_out_tr_temp)
            loss=criterion(outputs, targets)
            total_batch_loss += loss.item()

            if MSE == False:
                tr_correct += (torch.argmax(outputs, axis=1) == targets).sum().item()
                tr_total += len(targets)      
            #%%%%%%%%%%%add regularization%%%%%%%%%%%#
            for ind in range(n_features):
                count=1
                reg_loss = torch.tensor(0).float().to(device)
                for name, param in model.named_parameters():                        
                    if 'weight' in name and name.find('feature_nns.'+str(ind)) == 0: 
                        if  count == n_layer: 
                            reg_loss+=groupl1(param.T) 
                        count+=1
                loss += lbd*torch.sqrt(reg_loss)
            loss.backward()
            optimizer.step()
        
        if adaptive_lr:
            if (epoch+1)%10==0:
                scheduler.step()
            for g in optimizer.param_groups:
                wandb.log({"current_lr" : g['lr']})
        
        # evaluate net on validation set
        if test==True:
            model.eval()
            test_loss = 0
            test_correct, test_total = 0, 0
            with torch.no_grad():
                for idx2, (inputs, targets) in enumerate(testloader):
                    inputs, targets = inputs.to(device), targets.to(device)
                    outputs, f_out_te = model(inputs)[0], model(inputs)[1].detach().cpu().numpy()
                    test_loss += criterion(outputs, targets).item()
                    if MSE == False:
                        test_correct += (torch.argmax(outputs, axis=1) == targets).sum().item()
                        test_total += len(targets)
                test_loss = test_loss/(idx2+1)
                if (epoch+1)%10==0:
                    print(f"test loss : {test_loss}")
         
        # print train and val loss every 10 epochs
        if (epoch+1)%10==0:
            if MSE == False:
                print('epoch: ', epoch+1, 'Loss train: ', total_batch_loss/(idx1+1), 'ACC train: ',
                      100*(tr_correct/tr_total))
                print('Loss test: ', test_loss, 'ACC test: ', 100*(test_correct/test_total))
            else:
                print('epoch: ', epoch+1, 'Loss train: ', total_batch_loss/(idx1+1), 'Loss test: ',
                      test_loss)
        
        # log losses to wandb
        wandb.log({"Train Loss" : total_batch_loss/(idx1+1),
                           'Loss test: ' : test_loss})
        # save metrics to list for plotting
        epoch_train_loss_history.append(total_batch_loss/(idx1+1))
        epoch_test_loss_history.append(test_loss)
         
        # save model
        torch.save(model, save_dir+'model_'+run_id+'.pth') # wandb run id
        
        # early stopping
        #early_stopping(total_batch_loss, test_loss)
        
        #if early_stopping.early_stop:
        #  print("Early stopping at epoch:", epoch)
        #  break
    
    # show duration of training
    time_spent = round((time.time()-start)/60, 2)
    print(f"#################### Finished training in {time_spent} minutes ####################")
    
    wandb.finish()
    
    if MSE == False:
        return [100*(test_correct/test_total), test_loss, np.vstack(np.array(f_out_tr)), 
                np.vstack(np.array(f_out_te)), model]
    
    return [test_loss, np.vstack(np.array(f_out_tr)), np.vstack(np.array(f_out_te)), 
            model, epoch_train_loss_history, epoch_test_loss_history]

# compile model
def compile_model(lbd, lr, n_features, output_size, opti_name, device, seed=1, pyramid=False):
    """
    Select model architecture and compile the model
    Inputs:   lbd, n_features, output_size, seed -> int
              lr -> float
              pyramid -> bool
    """
    # start wandb run
    wandb.init(project="interpretable-ml", group="snam-studies")

    wandb.log({"learning_rate" : lr, "optimizer" : opti_name,
               "n_features" : n_features})
    # set seed
    torch.manual_seed(seed)
    np.random.seed(seed)
    
    # To-DO: implement transfer learning!
    # if loaded:
    #    model = model
    if pyramid == True: 
        model = NAM(n_features, output_size, PyramidNet, seed=1).to(device) #, MSE=MSE
        print(f"Training on: {device}")
    else: 
        model = NAM(n_features, output_size, SampleNet,seed=1).to(device) #, MSE=MSE
        print(f"Training on: {device}")

    if opti_name=='SGD':
        optimizer = torch.optim.SGD(model.parameters(), lr = lr)
    elif opti_name=='Adam': 
        optimizer = torch.optim.Adam(model.parameters(), lr = lr)
    else:
        print('wrong name. Pls enter one of them: SGD, ADAM, prox.')
    
    return model, optimizer


## Implement early stopping to prevent model from overfitting (TO-DO)
class EarlyStopping():
    
    def __init__(self, tolerance=10, min_delta=0.15):
        self.tolerance = tolerance
        self.min_delta = min_delta
        self.counter = 0
        self.early_stop = False

    def __call__(self, train_loss, validation_loss):
        if (validation_loss - train_loss) / train_loss > self.min_delta:
            self.counter +=1
            if self.counter >= self.tolerance:  
                self.early_stop = True