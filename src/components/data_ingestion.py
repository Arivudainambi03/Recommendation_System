import sys
import pandas as pd
from src.exception import CustomerException
from src.logger import logging
from dataclasses import dataclass
from surprise import Dataset, Reader
from surprise.model_selection import train_test_split

class DataIngestion:
    def __init__(self) -> None:
        pass
        
    def initial_data_ingestion(self):
        logging.info("Data ingestion started")

        try:
            # Read data
            df = pd.read_csv(r'notebook\data\Electronics_Dataset.csv', header=None)
            df.columns = ['user_id', 'prod_id', 'rating', 'timestamp']
            df.drop(columns=['timestamp'], inplace= True)

            # sort the 1000 above count alone
            rating_count = df.groupby(by = 'prod_id')['rating'].count()
            popular_products = rating_count[rating_count >= 1000].index
            rec_data = df[df['prod_id'].isin(popular_products)]

            # Convert DataFrame to Surprise Dataset
            reader = Reader(rating_scale=(1, 5))
            data = Dataset.load_from_df(rec_data, reader)

            # Split data into train and test
            trainset, testset = train_test_split(data, test_size=0.3, random_state=19)

            # Logging
            logging.info("Read and split the dataframe")
 
            return (trainset, testset, data)
        
        except Exception as e:
            raise CustomerException(e, sys)
        
# if __name__ == "__main__":
#     obj = DataIngestion()
#     train_data, test_data = obj.initial_data_ingestion()
#     logging.info("Train and Test Data are spilited.")
