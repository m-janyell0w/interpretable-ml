# evaluate a trained model and its outputs
from sklearn.metrics import mean_absolute_error, mean_squared_error
import torch
from tqdm import tqdm
import wandb



def count_selected_features(f_j):
    """
    Find the selected features, i.e. with non-zero shape function values.
    Input:   f_j -> np-3d-array
    Output:  
    """
    f_j = np.squeeze(f_j_tr)
    
    sum_f_j_list = []
    for col in range(f_j_tr.shape[-1]):
        sum_f_j = f_j_tr[:,col].sum()
        sum_f_j_list.append(sum_f_j)
    
    nonzero_features = [i for i in range(sum_f_j_list) if sum_f_j_list[i]!=0]

    
def predict_and_evaluate_model(model, testloader, run_id=None, wandb_log=False):
    """
    Predicts and evaluates the given model on given unseen data. 
    """
    
    f_out_te = []
    model.eval()
    #model.to('cpu')
    with torch.no_grad():
        for idx2, (inputs, targets) in tqdm(enumerate(testloader)):
            inputs, targets = inputs.to(device), targets.to(device).reshape(-1) 
            #.detach().cpu().numpy()
            outputs = model(inputs)[0].detach().cpu().numpy() 
            f_out_te_temp = model(inputs)[1].detach().cpu().numpy()
            f_out_te.append(f_out_te_temp)
            targets = targets.detach().cpu().numpy()
        
        # Calculate Test metrics: MAE, MSE, RMSE:
        rmse = np.sqrt(((targets - outputs)**2).mean())
        mae = mean_absolute_error(targets, outputs)
        mse = mean_squared_error(targets, outputs)
        
        if wandb_log:
            wandb.log({"rmse_test" : rmse, "mse_test" : mse, "mae_test" : mae})
    return rmse, mae, mse, outputs, targets, f_out_te