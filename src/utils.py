# some utility functions for e.g. data preprocessing
import pandas as pd

def data_splitter(data, end_of_training="1985-12-31", end_of_validation="1990-12-31"):
    """
    Splits time series data according to the dates provided.
    Inputs:   data -> pd.DataFrame
              end_of_training -> string
              end_of_validation -> string
    Outputs:  X_train, X_valid, X_test, y_train, y_valid, y_test -> pd.DataFrame
            
    """
    
    # define start of training
    start_of_training = data.index.get_level_values(0).min()
    
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