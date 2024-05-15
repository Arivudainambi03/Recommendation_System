import os 
import sys
import pandas as pd
from src.exception import CustomerException
from src.logger import logging
from dataclasses import dataclass
from surprise import Dataset, Reader
from surprise.model_selection import train_test_split

@dataclass
class DataIngestionConfig:
    train_data_path : str = os.path.join('artifacts', 'train.csv')
    test_data_path : str = os.path.join('artifacts', 'test.csv')
    raw_data_path :str = os.path.join('artifacts', 'data.csv')


class DataIngestion:
    def __init__(self) -> None:
        self.ingestion_cofig = DataIngestionConfig()

    def initial_data_ingestion(self):
        logging.info("Data ingestion started")
        try:
            df = pd.read_csv(r'notebook\data\Electronics_Dataset.csv', header= ['user_id', 'prod_id', 'rating', 'timestamp'])
            logging.info("Read the dataframe")

            os.makedirs(os.path.dirname(self.ingestion_cofig.train_data_path), exist_ok=True)

            df.to_csv(self.ingestion_cofig.raw_data_path, index = False, header=True)
            logging.info('Train test split initiated.')

            reader = Reader(rating_scale=(1, 5))
            reader_data = Dataset.load_from_df(df, reader)

            trainset, testset = train_test_split(reader_data, test_size = 0.3, random_state=19)

            trainset.to_csv(self.ingestion_cofig.train_data_path, index = False, header = True)
            logging.info("Train dataset got executed.")

            testset.to_csv(self.ingestion_cofig.test_data_path, index = False, header = True)
            logging.info("Test dataset got executed.")

            return (self.ingestion_cofig.train_data_path, self.ingestion_cofig.test_data_path)
        
        except Exception as e:
            raise CustomerException(e, sys)
        
if __name__ == "__main__":
    obj = DataIngestion()
    train_data, test_data = obj.initial_data_ingestion()

