# Utility functions such as data loading and processing
import pickle
import torch
from torch.utils.data import Dataset, DataLoader, TensorDataset

# small utilities like penalty and exU
def l1(x):
    """
    Computes l1 norm of a tensor.
    Inputs:   x -> torch.Tensor
    Outputs:  l1(x) -> torch.Tensor
    """
    return torch.sum(abs(x))

def ExU(x, w, b):
    return torch.matmul(torch.exp(w), x-b)

def groupl1(x):
    return torch.norm(x)

def sign(m):
    return torch.sign(m)

# convert data to 
def data_totensor(X_train, X_test, y_train, y_test, batch_size = 256, 
                  batch_size_test=1000,save=False, MSE=True):
    """
    Converts data sets to torch dataloaders, i.e. tensors.
    Inputs:   X_train, X_test, y_train, y_test -> np.ndarrays
              batch_size, batch_size_test -> int
              save, MSE-> bool
    Outputs:  trainloader, testloader -> torch.DataLoader
              D_out -> int
    
    """
    #%% separate data into three parts: train, validation(?) and test. 
    if MSE: 
        X_train_tensor = torch.tensor(X_train.values, dtype=torch.float32)
        y_train_tensor = torch.tensor(y_train.values, dtype=torch.float32).view(-1,1)
        X_test_tensor = torch.tensor(X_test.values, dtype=torch.float32)
        y_test_tensor = torch.tensor(y_test.values, dtype=torch.float32).view(-1,1)
        D_out = 1
    else:
        X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
        y_train_tensor = torch.tensor(y_train, dtype=torch.float32).long()
        X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
        y_test_tensor = torch.tensor(y_test, dtype=torch.float32).long()
        D_out = 2

    Train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
    trainloader = DataLoader(dataset=Train_dataset, batch_size=batch_size, shuffle=False)
    Test_dataset = TensorDataset(X_test_tensor, y_test_tensor)
    testloader = DataLoader(dataset=Test_dataset, batch_size=batch_size_test, shuffle=False)
    
    # TO-DO: if path does not exist yet...
    if save == True:
        f = open(path+"data/p2_data.pkl","wb")
        pickle.dump([trainloader, testloader, D_out], f)
        f.close()
    
    return trainloader, testloader, D_out