import os,sys
from src.logger import logging
from src.exception import CustomException
import pandas as pd
from sklearn.model_selection import train_test_split
from dataclasses import dataclass

from src.components.data_transformation import DataTrasformation
from src.components.data_transformation import DataTransformationConfig

@dataclass
class DataIngestionConfig:
    train_data_path : str=os.path.join('artifacts','train.csv')
    test_data_path : str=os.path.join('artifacts','test.csv')
    raw_data_path : str=os.path.join('artifacts','data.csv')


class dataingestion:
    def __init__(self):
        self.ingestion_config= DataIngestionConfig()

    def initiate_data_ingestion(self):
        logging.info('Enter the data ingestion methon or component')
        try:
            df = pd.read_csv('notebook\data\StudentsPerformance.csv')
            logging.info('imported the csv')
            os.makedirs(os.path.dirname(self.ingestion_config.test_data_path),exist_ok=True)
            df.to_csv(self.ingestion_config.raw_data_path,index=False,header=True)
            logging.info('data has been added')
            train_set,test_set = train_test_split(df,test_size=0.2,random_state=30)
            train_set.to_csv(self.ingestion_config.train_data_path,index=False,header=True)
            test_set.to_csv(self.ingestion_config.test_data_path,index=False,header=True)
            logging.info('train and test ingestion has been completed')
            return(
                self.ingestion_config.train_data_path,
                self.ingestion_config.test_data_path
                
            )
            

        except Exception as e:
            raise CustomException(e,sys)
        

if __name__=='__main__':
    obj=dataingestion()
    train_data,test_data=obj.initiate_data_ingestion()

    data_transformation = DataTrasformation()
    data_transformation.initiate_data_trasformation(train_data,test_data)