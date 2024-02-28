import sys
import os
from dataclasses import dataclass
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from src.exception import CustomException
from src.logger import logging
from src.utils import save_object


@dataclass
class DataTransformationConfig:
    preprocessor_file_path = os.path.join('artifacts', 'preprocess', 'preprocessor.pkl')


class DataTransformation:
    def __init__(self):
        self.data_transformation_config = DataTransformationConfig()

    def get_data_transformer_object(self):
        """
            This function is responsible for data transformation
        """

        try:
            numerical_columns = ["writing score", "reading score"]
            categorical_columns = [
                "gender",
                "race/ethnicity", 
                "parental level of education",
                "lunch",
                "test preparation course"
            ]
            logging.info('1')

            numerical_pipeline = Pipeline(
                steps=[
                    ("imputer", SimpleImputer(strategy="median")),
                    ("scaler", StandardScaler(with_mean=False))
                ]
            )

            logging.info('2')

            categorical_pipeline = Pipeline(
                steps=[
                    ("imputer", SimpleImputer(strategy="most_frequent")),
                    ('one_hot_encoder', OneHotEncoder()),
                    ('scaler', StandardScaler(with_mean=False))
                ]
            )

            logging.info('3')


            logging.info(f"Categorical  columns are {categorical_columns}")
            logging.info(f"Numerical columns are {numerical_columns}")

            column_transformer = ColumnTransformer(
                [
                    ("numerical_pipeline", numerical_pipeline, numerical_columns),
                    ("categorical_pipelines", categorical_pipeline, categorical_columns)
                ]
            )

            return column_transformer

        except Exception as e:
            CustomException(e, sys)


    def initiate_data_transformation(self, train_path, test_path):
        # try:
        train_df = pd.read_csv(train_path)
        test_df = pd.read_csv(test_path)

        logging.info("Read train and test data completed")
        
        # Transforming the dataset using the defined pipeline
        column_transformer = self.get_data_transformer_object()
        
        target_column_name = "math score"
        numerical_columns = ["writing score", "reading score"]

        input_feature_train_df = train_df.drop([target_column_name], axis=1)
        target_feature_train_df = train_df[target_column_name]

        input_feature_test_df = test_df.drop([target_column_name], axis=1)
        target_feature_test_df = test_df[target_column_name]

        logging.info("Applying preprocesssing on training dataframe and testing dataframe")

        input_feature_train_arr = column_transformer.fit_transform(input_feature_train_df)
        input_feature_test_arr = column_transformer.transform(input_feature_test_df)


        train_arr = np.c_[
            input_feature_train_arr, np.array(target_feature_train_df)
        ]

        test_arr = np.c_[
            input_feature_test_arr, np.array(target_feature_test_df)
        ]

        logging.info("Saved preprocessing")

        save_object(
            file_path = self.data_transformation_config.preprocessor_file_path,
            obj = column_transformer
        )


        return (
            train_arr,
            test_arr,
            self.data_transformation_config.preprocessor_file_path
        )

        # except Exception as e:
        #     CustomException(e, sys)




