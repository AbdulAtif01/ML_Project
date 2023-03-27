import sys,os
from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder,StandardScaler

from src.exception import CustomException
from src.logger import logging

class DataTransformationConfig:
    preprocesser_obj_file_path=os.path.join('artifacts','preprocesser.pkl')

class DataTrasformation:
    def __init__(self) -> None:
        self.DataTransformationConfig=DataTransformationConfig()

    def get_data_transformer_obj (self):
        try:
            logging.info('Transforing has been started')
            df = pd.read_csv('artifacts\data.csv')
            numeric_features = [feature for feature in df.columns if df[feature].dtype != 'O']
            categorical_features = [feature for feature in df.columns if df[feature].dtype == 'O']
            num_pipeline =Pipeline(
                steps=[
                ('imputer',SimpleImputer(strategy='median')),
                ('sclaer',StandardScaler())
                ]
            )
            logging.info('numerical data transformation completed')
            categorical_pipeline=Pipeline(
                steps=[
                ('imputer',SimpleImputer(strategy="most_frequent")),
                ('one_hot_encoder',OneHotEncoder()),
                ('sclar',StandardScaler())

                ]
            )
            logging.info('categorical data tranformation completed')
            logging.info(f'cat_feature:{categorical_features}')
            logging.info(f'num_feature:{numeric_features}')

            preprocesser = ColumnTransformer(
                ['num_pipeline',num_pipeline,numeric_features],
                ['cat_pipeline',categorical_pipeline,categorical_features]
            )
            

            return preprocesser

        except Exception as e:
            raise CustomException(e,sys)
        

    def initiate_data_trasformation(self,train_path,test_path):
        try:
            train_df=pd.read_csv(train_path)
            test_df=pd.read_csv(test_path)
            logging.info('Reading train and test data')

            logging.info('obtaining preprocessing object')
            preprocesser_obj = self.get_data_transformer_obj()

            target_column_name='math score'
            numerical_columns = ['writing score','reading score']

            input_feature_test_df = test_df.drop(columns=[target_column_name],axis = 1)
            target_feature_test_df = test_df[target_column_name]

            logging.info(f"applying preprocessing object on training dataframe and testing")

            input_feature_train_arr = preprocesser_obj.fit_transform(input_feature_test_df)
            input_feature_test_arr = preprocesser_obj.fit(input_feature_test_df)

            train_arr = np.c_[input_feature_train_arr,np.array(input_feature_train_arr)]
            test_arr = np.c_[input_feature_test_arr,np.array(input_feature_test_arr)]

            logging.info('saved preprocessing object')

            

            return(
                train_arr,
                test_arr,
                self.DataTransformationConfig.preprocesser_obj_file_path
            )


        except:
            pass
