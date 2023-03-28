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
from src.utils import save_object

class DataTransformationConfig:
    preprocesser_obj_file_path=os.path.join('artifacts','preprocesser.pkl')

class DataTrasformation:
    def __init__(self):
        self.DataTransformationConfig=DataTransformationConfig()

    def get_data_transformer_obj (self):
        try:
            logging.info('Transforing has been started')
            numerical_columns = ['reading score', 'writing score']
            categorical_columns = [
                "gender",
                "race/ethnicity",
                "parental level of education",
                "lunch",
                "test preparation course",
            ]
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
                ('sclar',StandardScaler(with_mean=False))

                ]
            )
            logging.info('categorical data tranformation completed')
            logging.info(f'cat_feature:{categorical_columns}')
            logging.info(f'num_feature:{numerical_columns}')

            preprocesser = ColumnTransformer(
                [('num_pipeline',num_pipeline,numerical_columns),
                ('cat_pipeline',categorical_pipeline,categorical_columns)]
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

            input_feature_train_df=train_df.drop(columns=[target_column_name],axis=1)
            target_feature_train_df=train_df[target_column_name]

            input_feature_test_df = test_df.drop(columns=[target_column_name],axis = 1)
            target_feature_test_df = test_df[target_column_name]

            logging.info(f"applying preprocessing object on training dataframe and testing")

            input_feature_train_arr = preprocesser_obj.fit_transform(input_feature_train_df)
            input_feature_test_arr = preprocesser_obj.transform(input_feature_test_df)

            train_arr = np.c_[input_feature_train_arr,np.array(input_feature_train_arr)]
            test_arr = np.c_[input_feature_test_arr,np.array(input_feature_test_arr)]

            logging.info('saved preprocessing object')

            save_object(
                file_path =self.DataTransformationConfig.preprocesser_obj_file_path,
                obj = preprocesser_obj
            )

            

            return(
                train_arr,
                test_arr,
                self.DataTransformationConfig.preprocesser_obj_file_path
            )


        except Exception as e:
            raise CustomException (e,sys)
            