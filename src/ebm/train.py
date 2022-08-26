# import basic libraries
from datetime import datetime
from interpret.glassbox import ExplainableBoostingRegressor
from interpret import show
import numpy as np
import pandas as pd
import pickle
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.preprocessing import StandardScaler
import wandb

from src.utils import data_loader, time_series_splitter
   

class EbmPipeline():
    """
    Define pipeline class which runs a model training and stores training artifacts
    and other things. It comes with a number of default training parameters like e.g. 
    train, val and test periods.
    """
    def __init__(self, root_path="./data/", data_dir="", pickle=False):
        # training parameters
        self.root_path: str = root_path
        self.data_dir: str = data_dir
        self.pickle: bool = pickle
        # self.start_of_training: str = "1965-01-01"
        # self.end_of_training: str = "1985-12-31",
        # self.end_of_validation: str ="1990-12-31" 
        self.tune: bool = False,
        self.param_dict: dict = {}
        self.model = ExplainableBoostingRegressor(random_state=0)
    
    def run(self, feature_selector=[]) -> dict:
        """
        Runs the pipeline considering experiment tracking, data loading, splitting, 
        modelling and evaluation. 
        """
        print("========== Setup training ============")
        
        # initialize wandb for tracking
        wandb.init(project="interpretable-ml", group="ebm-studies")
        
        # start timer for total run time
        start_time = datetime.now()

        # load and split data
        print("========== Load data ============")
        data = data_loader(self.root_path, self.data_dir, self.pickle, feature_selector)
        data = normalize_target(data)
        
        print("========== Split data ============")
        X_train, X_valid, X_test, y_train, y_valid, y_test = \
        time_series_splitter(data,
                             #self.start_of_training, self.end_of_training,
                             #self.end_of_validation
                            )
        
        # train model
        print("========== Start training ============")
        self.train_model(X_train, y_train)
        print("========== Finished training ============")
        
        # give global model explanations using built-in viz
        #self.visualize_ebm_global()
        
        print("========== Evaluate model ============")
        # evaluate model
        y_pred = self.model.predict(X_valid)
        #y_pred_test = self.model.predict(X_test)
        is_r_squared = self.validate_model(X_train, y_train)
        oos_r_squared = self.validate_model(X_valid, y_valid)
        rmse_test = self.rmse(y_valid, y_pred)
        mse_test = mean_squared_error(y_valid, y_pred)
        mae_test = mean_absolute_error(y_valid, y_pred)
        
        # other interesting attributes
        model_params = self.model.get_params()
        n_features = X_train.shape[1]
        execution_time = round((datetime.now() - start_time).total_seconds(),2)
        
        print("========== Finish up training pipeline ============")
        # log metrics
        wandb.log({"mse" : mse_test, 
                   "rmse" : rmse_test, 
                   "mae" : mae_test, 
                   "n_features": n_features,
                   "time_spent" : execution_time, 
                   "Model Params" : model_params})
        wandb.finish()
        
        pipeline_artifacts = {
            "execution_time" : execution_time,
            "r-squared" : [is_score, oos_score], 
            "rmse_test" : rmse_test, 
            "mse_test" : mse_test, 
            "mae_test" : mae_test,
            "model_params" : model_params, 
            "test_set" : [X_test, y_test], 
            "valid_set" : [X_valid, y_valid], 
            "y_pred_val" : y_pred,
            #"y_pred_test" : y_pred_test
        }
        
        return pipeline_artifacts

        
    def train_model(self, X_train, y_train):
        """
        Train model on possibly given hyperparameters.
        """

        if self.tune:
            print("model is being tuned")
            self.model.set_params(**self.param_dict)
        self.model = self.model.fit(X_train, y_train)
        #return self.model

    def validate_model(self, X_valid, y_valid):
        """
        Returns R-squared of given model on the specified data set.
        """
        r_squared = self.model.score(X_valid, y_valid) # validation set R²
        return r_squared
        
    def rmse(self, y_test, y_pred):
        """
        Returns RMSE for ground truth and prediction vectors.
        """
        rmse_test = np.sqrt(((self.y_test - self.y_pred)**2).mean())  
        # np.sqrt(((predictions - targets) ** 2).mean())
        return rmse_test
        
    def visualize_ebm_global(self):
        """
        Visualize global explanation of given model
        """
        ebm_global = self.model.explain_global()
        show(ebm_global)
        return ebm_global

def normalize_target(data):
    """
    Normalizes target variable y of a dataset.
    """
    scaler = StandardScaler()
    target = data.TARGET.values.reshape(-1, 1)
    #print(target.shape)
    data['TARGET'] = scaler.fit_transform(target)

    return data
            
def save_model(model, model_dir='./models/ebm/', run_id='00'):
    """
    Save model params to disk
    """
    
    param_dict = model.get_params(deep=False)
    with open(f'{model_dir}ebm_{run_id}.pkl', 'wb') as f:
        pickle.dump(param_dict, f)
        
        
def load_model(model_dir='./models/ebm/', run_id='00'):
    """
    Load saved model from indicated directory.
    """
    
    with open(f"{model_dir}ebm_{run_id}.pkl","rb") as f:
        model = pickle.load(f)
    return model

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