# some utility functions for e.g. data preprocessing
import pandas as pd
import pickle
from sklearn.preprocessing import StandardScaler


def data_loader(path, data_dir, pickle=False, feature_selector=[]):
    """
    Loads time series data from indicated directory.
    Inputs:   path: string
              data_dir: string
              pickle: bool
              feature_selector: list
    Outputs:  data: pd.DataFrame
    
    """
    # parse dates to datetime in format YY-MM-DD before loading
    dateparse = lambda x: pd.to_datetime(x, format='%Y-%m-%d', errors='coerce')
    
    if pickle:
        data = pd.read_pickle(path + data_dir)
    else:
        data = pd.read_csv(path + data_dir, 
                         parse_dates=['date'],
                         index_col=['date', 'permno'], 
                         date_parser=dateparse, skipinitialspace=True)
    
    if len(feature_selector) > 0:
        data = data.loc[:, feature_selector]
    
    return data
    

def time_series_splitter(data, start_of_training = "1965-01-01",
                         end_of_training="1985-12-31", end_of_validation="1990-12-31"):
    """
    Splits time series data according to the dates provided.
    Inputs:   data -> pd.DataFrame
              end_of_training -> string
              end_of_validation -> string
    Outputs:  X_train, X_valid, X_test, y_train, y_valid, y_test -> pd.DataFrame
            
    """
    #start_of_training = data.index.get_level_values(0).min()
    
    # isolate y from data set
    X = data.iloc[:,:-1]
    y = data.TARGET

    # slice to required size
    X_train = X.loc[pd.IndexSlice[start_of_training:end_of_training,], :]
    y_train = y.loc[pd.IndexSlice[start_of_training:end_of_training,]]
    X_valid = X.loc[pd.IndexSlice[end_of_training:end_of_validation,], :]
    y_valid = y.loc[pd.IndexSlice[end_of_training:end_of_validation,]]
    X_test = X.loc[pd.IndexSlice[end_of_validation:,], :]
    y_test = y.loc[pd.IndexSlice[end_of_validation:,]]
    
    return X_train, X_valid, X_test, y_train, y_valid, y_test

def normalize_target(data):
    """
    Normalizes target variable y of a dataset.
    """
    scaler = StandardScaler()
    target = data.TARGET.values.reshape(-1, 1)
    #print(target.shape)
    data['TARGET'] = scaler.fit_transform(target)

    return data

# TO-DO; generalize it
# def save_model(model, model_dir='./models/ebm/', run_id='00'):
#     """
#     Save model params to disk
#     """
    
#     param_dict = model.get_params(deep=False)
#     with open(f'{model_dir}ebm_{run_id}.pkl', 'wb') as f:
#         pickle.dump(param_dict, f)
        
        
# def load_model(model_dir='./models/ebm/', run_id='00'):
#     """
#     Load saved model from indicated directory.
#     """
    
#     with open(f"{model_dir}ebm_{run_id}.pkl","rb") as f:
#         model = pickle.load(f)
#     return model

def denormalize(data, scaler_path, dataframe=True):
    """
    Inverse transforms the given dataset using the scaler that was used
    to normalize it -> back to original scale.
    
    Parameters
    -----------
        data (pd.DataFrame): The dataset to transform
        scaler_path (str): The exact path to the scaler to use
        dataframe (bool): Whether to convert the data to a dataframe
    
    Returns
    ----------
        data_denorm (pd.DataFrame): The denormalized dataset 
    """
    
    with open(f"{scaler_path}", "rb") as f:
        scaler = pickle.load(f)
    
    data_denorm = scaler.inverse_transform(data)
    
    if dataframe:
        data_denorm = pd.DataFrame(columns=data.columns)
    
    return data_denorm