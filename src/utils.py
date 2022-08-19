# some utility functions for e.g. data preprocessing
import pandas as pd


def data_loader(path, data_dir, pickle=False, feature_selector=None):
    """
    Loads time series data from indicated directory.
    Inputs:   path -> string
              data_dir -> string
              pickle -> bool
              feature_selector -> list
    Outputs:  data -> pd.DataFrame
    
    """
    # parse dates to datetime in format YY-MM-DD before loading
    dateparse = lambda x: pd.to_datetime(x, format='%Y-%m-%d', errors='coerce')
    
    if pickle:
      data = pd.read_pickle(path + data_dir)
    else:
      data = pd.read_csv(path + data_dir, parse_dates=['date'], 
                       index_col=['date', 'permno'], date_parser=dateparse, skipinitialspace=True)
    if feature_selector is not None:
      data = data.loc[:, feature_selector]
    
    return data
    

def time_series_splitter(data, start_of_training = data.index.get_level_values(0).min(),
                         end_of_training="1985-12-31", end_of_validation="1990-12-31"):
    """
    Splits time series data according to the dates provided.
    Inputs:   data -> pd.DataFrame
              end_of_training -> string
              end_of_validation -> string
    Outputs:  X_train, X_valid, X_test, y_train, y_valid, y_test -> pd.DataFrame
            
    """
    
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