# import basic libraries
from datetime import datetime
from interpret.glassbox import ExplainableBoostingRegressor
from interpret import show
import numpy as np
import pandas as pd
import pickle
from sklearn.metrics import mean_absolute_error, mean_squared_error
import wandb

from src.utils import data_loader, time_series_splitter
   

class EbmPipeline():
    """
    Define pipeline class which runs a model training and stores training artifacts
    and other things.
    """
    def __init__(self, root_path="./data/", data_dir="", pickle=False):
        # training parameters
        self.root_path: str = root_path
        self.data_dir: str = data_dir
        self.pickle: bool = pickle
        self.start_of_training: str = "1965-01-01"
        self.end_of_training: str = "1985-12-31",
        self.end_of_validation: str ="1990-12-31" 
        self.tune: bool = False, 
        self.feature_selection: list = [],
        self.param_dict: dict = {}
        self.model = ExplainableBoostingRegressor(random_state=0)
    
        # train artifacts
        self.data = None
        self.X_train, self.X_valid, self.X_test = None, None, None
        self.y_train, self.y_valid, self.y_test = None, None, None
        self.execution_time = None
        self.is_r_squared = None
        self.oos_r_squared = None
        self.rmse_test = None
        self.mse_test = None
        self.mae_test = None
        self.model_params = None
        self.y_pred = None
        self.n_features = None
    
    def run(self):
        """
        Runs the pipeline considering experiment tracking, data loading, splitting, modelling and
        evaluation.

        """

        # initialize wandb for tracking
        wandb.init(project="interpretable-ml", group="ebm-studies")
        # start timer for total run time
        start_time = datetime.now()

        # load and split data
        self.data = data_loader(root_path, data_dir, pickle, feature_selection)
        self.data = normalize_target(data)
        self.X_train, self.X_valid, self.X_test, self.y_train, self.y_valid, self.y_test = \
        time_series_splitter(self.data, self.start_of_training, self.end_of_training, self.end_of_validation)

        # train model
        self.model = train_model(model, X_train, y_train, tune, param_dict)

        # give global model explanations using built-in viz
        visualize_ebm_global(fitted_model)

        # get interesting metrics and attributes
        self.model_params = fitted_model.get_params()
        self.is_r_squared = validate_model(model, X_train, y_train)
        self.oos_r_squared = validate_model(model, X_valid, y_valid)
        self.rmse_test = rmse(y_valid,y_pred)
        self.mse_test = mean_squared_error(y_valid, y_pred)
        self.mae_test = mean_absolute_error(y_valid,y_pred)

        self.y_pred = fitted_model.predict(X_valid)
        self.n_features = X_train.shape[1]
        self.execution_time = round((datetime.now() - start_time).total_seconds(),2)

        # log metrics
        wandb.log({"mse" : mse, "rmse" : rmse_test, "mae" : mae, "n_features": n_features, 
                   "time_spent" : execution_time, "Model Params" : params})
        wandb.finish()

        #return execution_time, is_score, oos_score, rmse_test, mse_test, mae_test,
        #        params, X_valid, y_valid, y_pred, fitted_model

        
        def train_model(self):
            """
            Train model on possibly given hyperparameters.
            """

            if self.tune:
                print("model is being tuned")
                self.model.set_params(**self.param_dict)
            self.model.fit(self.X_train, self.y_train)
            #return self.model
        
        def validate_model(self):
            """
            Returns R-squared of given model on the specified data set.
            """
            self.model.score(self.X_valid, self.y_valid) # validation set R²

        def rmse(self):
            """
            Returns RMSE for ground truth and prediction vectors.
            """
            self.rmse_test = np.sqrt(((self.y_test - self.y_pred)**2).mean())  
            # np.sqrt(((predictions - targets) ** 2).mean())

        def visualize_ebm_global(self):
            """
            Visualize global explanation of given model
            """
            ebm_global = self.model.explain_global()
            show(ebm_global) 

            
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
        model = pickle.load(model_artifacts, f)
    return model