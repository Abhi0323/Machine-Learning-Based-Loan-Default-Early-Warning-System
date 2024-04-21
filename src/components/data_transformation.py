import os
import sys
from src.logger import logging
from src.exception import CustomException
from src.utils import save_object
from src.utils import convert_target_variable

import numpy as np 
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder,StandardScaler
from dataclasses import dataclass

@dataclass
class DataTransformetaionConfig:
    data_processor_obj_file_path= os.path.join("artificats", "processor.pkl")

class DataTransformetaion:
    def __init__(self):
        self.data_transformetaion_config = DataTransformetaionConfig()

    def get_transformed_data(self):
        try:
            ''' This function is responsible for transforming data '''
            
            num_features = ["Age", "Income", "Amount", "Status", "Percent_income", "Cred_length"]
            impute_median_features = ["Emp_length"]
            impute_mean_features = ["Rate"]
            cat_features = ["Home", "Intent"]

            num_pipeline = Pipeline([
                ("scaler", StandardScaler())
            ])
            
            impute_median_pipeline = Pipeline([
                ("imputer_median", SimpleImputer(strategy='median')),
                ("scaler", StandardScaler())
            ])

            impute_mean_pipeline = Pipeline([
                ("imputer_mean", SimpleImputer(strategy='mean')),
                ("scaler", StandardScaler())
            ])

            cat_pipeline = Pipeline(steps=[
                ("onehot", OneHotEncoder()),
                ('scaler', StandardScaler(with_mean=False))
            ])

            preprocessor = ColumnTransformer(transformers=[
                ("num", num_pipeline, num_features),
                ("impute_median", impute_median_pipeline, impute_median_features),
                ("impute_mean", impute_mean_pipeline, impute_mean_features),
                ("cat", cat_pipeline, cat_features)
            ])
            logging.info("Encoding and Imputation is implemented")

            return preprocessor
        except Exception as e:
            raise CustomException(e,sys)
        
    def initiate_transfromation(self, train_path, test_path):
        try: 
            
            train_df = pd.read_csv(train_path)
            test_df = pd.read_csv(test_path)

            logging.info("Train and Test data read successfully")

            logging.info("Obtaining preprocessing object")

            preprocess_obj = self.get_transformed_data()

            train_df['Default'] = convert_target_variable(train_df['Default'])
            test_df['Default'] = convert_target_variable(test_df['Default'])
            logging.info("Converted target variable to binary format")

            target_variable = ['Default']
            not_imp_variables = ['Id']

            input_features_train_df = train_df.drop(columns= target_variable + not_imp_variables, axis=1)
            target_train_df = train_df[target_variable]

            input_features_test_df = test_df.drop(columns= target_variable + not_imp_variables, axis=1)
            target_test_df = test_df[target_variable]

            logging.info(
                    f"Applying preprocessing obj on train df and test df"
                )
            input_features_train_df_array = preprocess_obj.fit_transform(input_features_train_df)
            input_features_test_df_array = preprocess_obj.transform(input_features_test_df)

            logging.info(
                    f"Applied preprocessing obj on train df and test df"
                )
            
            train_array = np.c_[input_features_train_df_array, np.array(target_train_df)]
            test_array = np.c_[input_features_test_df_array, np.array(target_test_df)]
            
            logging.info(
                    f"Saving preprocessing object"
                )
            
            save_object(file_path=self.data_transformetaion_config.data_processor_obj_file_path, obj= preprocess_obj)

            logging.info(
                    f"Saved preprocessing object"
                )
            
            return (
                train_array, 
                test_array, 
                self.data_transformetaion_config.data_processor_obj_file_path
                )
        except Exception as e:
            raise CustomException(e,sys)
            

