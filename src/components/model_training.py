import os
import sys
from src.logger import logging
from dataclasses import dataclass
from src.exception import CustomerException
from data_ingestion import DataIngestion
from surprise import NormalPredictor, KNNBaseline, SVD
from src.utils import ModelUtils


@dataclass
class ModelTrainerConfig:
    trained_model_file_path=os.path.join("artifacts","model.pkl")

class ModelTrainer:
    def __init__(self, trainset, testset, data):
        self.model_config = ModelTrainerConfig()
        self.trainset = trainset
        self.testset = testset
        self.data = data

    def train_and_evaluate(self):
        logging.info("training and evaluation is started.")

        try:
            # Normal Predictor
            algo = NormalPredictor()
            train_rmse, test_rmse = ModelUtils.evaluate_model(algo, self.trainset, self.testset)

            best_rmse = test_rmse
            best_algo = algo

            # KNN Baseline
            sim_option = {'name': "cosine", 'user_based': False}
            algo_knn = KNNBaseline(sim_options=sim_option)
            train_rmse, test_rmse = ModelUtils.evaluate_model(algo_knn, self.trainset, self.testset)
            
            if test_rmse < best_rmse:
                best_rmse = test_rmse
                best_algo = algo_knn

            # Hyperparameter tuning for KNN Baseline
            param_grid = {
                          'k': [10, 50, 100],
                          'sim_options': {'name': ['msd', 'cosine'],
                                          'user_based': [False]}
                          }
            
            best_rmse_tuned, best_params = ModelUtils.hyperparameter_tuning(KNNBaseline, param_grid, self.data)
            
            if best_rmse_tuned < best_rmse:
                best_rmse = best_rmse_tuned
                best_algo = KNNBaseline(sim_options=best_params['sim_options'], k=best_params['k'])

            # SVD
            svd_algo = SVD()
            train_rmse, test_rmse = ModelUtils.evaluate_model(svd_algo, self.trainset, self.testset)
            
            if test_rmse < best_rmse:
                best_rmse = test_rmse
                best_algo = svd_algo

            # Hyperparameter tuning for SVD
            param_grid = {"n_epochs": [5, 10, 15],
                          "lr_all": [0.002, 0.005, 0.007],
                          "reg_all": [0.4, 0.6]}
            
            best_rmse_tuned, best_params = ModelUtils.hyperparameter_tuning(SVD, param_grid, self.data)
            
            if best_rmse_tuned < best_rmse:
                best_rmse = best_rmse_tuned
                best_algo = SVD(n_epochs=best_params['n_epochs'], lr_all=best_params['lr_all'], reg_all=best_params['reg_all'])

            logging.info("Best found model on both training and testing dataset")

            # Save best model
            ModelUtils.save_model(algo = best_algo,
                                  filepath=self.model_config.trained_model_file_path)
            
            logging.info("Best model saved as 'model.pkl'")

        except CustomerException as e:
            raise CustomerException(e, sys)

if __name__ == "__main__":

    # Load data and split into train and test
    obj = DataIngestion()
    trainset, testset, data = obj.initial_data_ingestion()
    logging.info("Train and Test Data are split.")

    # Model Training
    trainer = ModelTrainer(trainset, testset, data)
    trainer.train_and_evaluate()
    logging.info("Training completed.")