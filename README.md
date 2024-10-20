# Machine Learning Project
The Math Score Prediction project is a comprehensive machine learning application designed to predict students' math scores based on various input features, including writing scores, reading scores, demographic information, and educational background. This project employs a robust machine learning pipeline that encompasses data ingestion, preprocessing, model training, and prediction.<br>

Key Features:<br>
Data Ingestion: Efficiently loads and splits the dataset into training and testing sets, ensuring a well-structured foundation for model training.<br>
Data Transformation: Preprocesses the input data by handling missing values, scaling numerical features, and encoding categorical variables, which enhances the model's performance.<br>
Model Training: Implements various regression algorithms, including Random Forest, Decision Tree, Gradient Boosting, and more, to find the best-performing model through hyperparameter tuning.<br>
Prediction Pipeline: Provides an intuitive interface for making predictions based on user inputs, enabling easy interaction with the trained model.<br>
Logging and Error Handling: Comprehensive logging and custom exception handling ensure that any issues during execution are well-documented, making debugging easier.<br>
Technologies Used:<br>
The project leverages popular libraries and frameworks such as:<br>

Pandas: For data manipulation and analysis.<br>
NumPy: For numerical computations and array manipulations.<br>
Seaborn & Matplotlib: For data visualization and exploratory data analysis.<br>
Scikit-learn: For implementing machine learning algorithms and tools.<br>
CatBoost & XGBoost: For advanced gradient boosting techniques.<br>
Flask: For building a web application interface for model predictions.<br>
Dill: For serialization of Python objects.<br>
Purpose:<br>
The primary goal of this project is to provide insights into the factors influencing students' performance in math, enabling educators and stakeholders to make informed decisions about curriculum improvements, personalized learning strategies, and resource allocation.<br>

## Overview
This project implements a machine learning pipeline to predict math scores based on various input features such as writing scores, reading scores, and demographic information. The pipeline includes data ingestion, transformation, model training, and prediction functionalities.

## Project Structure
ml_project/ ├── src/ │ ├── init.py │ ├── exception.py # Custom exception handling │ ├── logger.py # Logging configuration │ ├── pipeline/ │ │ ├── init.py │ │ ├── train_pipeline.py # Orchestrates the training process │ │ ├── predict_pipeline.py # Handles predictions │ ├── components/ │ │ ├── init.py │ │ ├── data_ingestion.py # Data loading and splitting │ │ ├── data_transformation.py # Data preprocessing and transformation │ │ └── model_trainer.py # Model training and evaluation │ ├── utils.py # Utility functions for saving/loading objects and model evaluation └── main.py # Entry point for the application


## Installation
1. Clone the repository:
```bash
   git clone https://github.com/yourusername/ml_project.git
   cd ml_project
   ```

2. Create a virtual environment and activate it:
```bash
conda create -n ml_project_env python=3.8 -y
conda activate ml_project_env
```

3. Install required packages:
```bash
pip install -r requirements.txt
```

## Usage
### Training the Model
To train the model, run the following command:
```bash
python src/pipeline/train_pipeline.py
```
This will ingest the data, preprocess it, train the models, and save the best-performing model and preprocessor.

### Making Predictions
To make predictions with the trained model, you can use the predict_pipeline.py. Here's how you can run it:
```bash
python src/pipeline/predict_pipeline.py
```
Ensure you provide the required input features through your web application or command line.

## Logging
Logs are stored in the logs directory, which is created automatically. The log files are named with timestamps for easy tracking.

## Exception Handling
Custom exceptions are defined in exception.py to provide meaningful error messages throughout the application.

## Contributing
Contributions are welcome! Please feel free to open an issue or submit a pull request.

## Acknowledgements
scikit-learn for machine learning algorithms and tools.<br> 
Pandas for data manipulation and analysis.<br> 
NumPy for numerical computations.<br> 
Seaborn for statistical data visualization.<br> 
Matplotlib for creating static, animated, and interactive visualizations.<br> 
CatBoost for gradient boosting on categorical features.<br> 
XGBoost for optimized gradient boosting framework.<br> 
Flask for building the web application.<br> 
Dill for object serialization.<br> 