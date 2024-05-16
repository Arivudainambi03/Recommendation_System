import os
import sys
import pickle
import pandas as pd
from surprise import accuracy, Dataset, Reader
from src.logger import logging
from src.exception import CustomerException
from surprise.model_selection import GridSearchCV

class ModelUtils:

    @staticmethod
    def evaluate_model(algo, trainset, testset):
        try:
            algo.fit(trainset)
            train_pred = algo.test(trainset.build_testset())
            test_pred = algo.test(testset)

            train_rmse = accuracy.rmse(train_pred)
            test_rmse = accuracy.rmse(test_pred)
            logging.info("Model Evaluation executed.")

            return train_rmse, test_rmse
        
        except CustomerException as e:
            raise CustomerException(e, sys)

    @staticmethod
    def hyperparameter_tuning(algo_class, param_grid, trainset):
        try:
            gs = GridSearchCV(algo_class, param_grid=param_grid, measures=['rmse'], cv=3)
            gs.fit(trainset)
            logging.info("Hyperparameter tuning executed.")
            return gs.best_score['rmse'], gs.best_params['rmse']
        except Exception as e:
            raise CustomerException(e, sys)
    
    @staticmethod
    def save_model(algo, filepath):
        try:
            dir_path = os.path.dirname(filepath)

            os.makedirs(dir_path, exist_ok=True)

            with open(filepath, 'wb') as f:
                pickle.dump(algo, f)
                logging.info("model saved sucessfully.")

        except CustomerException as e:
            raise CustomerException(e, sys)
        
    @staticmethod
    def load_model(filepath):
        try:
            with open(filepath, 'rb') as f:
                return pickle.load(f)
        
        except CustomerException as e:
            raise CustomerException(e, sys)