import os
import sys
from src.exception import CustomException
from src.logger import logging
from src.components.data_ingestion import DataIngestion
from src.components.data_transformation import DataTransformation
from src.components.model_trainer import ModelTrainer

class TrainerPipeline:
    def __init__(self):
        self.data_ingestion = DataIngestion()
        self.data_transformation = DataTransformation()
        self.model_trainer = ModelTrainer()

    def run_pipeline(self):
        try:
            # Data Ingestion
            train_data, test_data = self.data_ingestion.initiate_data_ingestion()
            logging.info("Data ingestion completed.")

            # Data Transformation
            train_arr, test_arr, _ = self.data_transformation.initiate_data_transformation(train_data, test_data)
            logging.info("Data transformation completed.")

            # Model Training
            r2_square = self.model_trainer.initiate_model_trainer(train_arr, test_arr)
            logging.info(f"Model training completed with R² score: {r2_square}")

            return r2_square

        except Exception as e:
            raise CustomException(e, sys)

if __name__ == "__main__":
    pipeline = TrainerPipeline()
    r2_score = pipeline.run_pipeline()
    print(f"Final R² Score: {r2_score}")
