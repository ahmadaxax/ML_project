from flask import Flask, request, render_template, redirect, url_for, flash
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from src.pipeline.predict_pipeline import CustomData, PredictPipeline
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)

application = Flask(__name__)
app = application
app.secret_key = 'q'  

# Route for home page
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predictdata', methods=['GET', 'POST'])
def predict_datapoint():
    if request.method == 'POST':
        # Collect and validate form data
        try:
            data = CustomData(
                gender=request.form['gender'],
                race_ethnicity=request.form['ethnicity'],
                parental_level_of_education=request.form['parental_level_of_education'],
                lunch=request.form['lunch'],
                test_preparation_course=request.form['test_preparation_course'],
                reading_score=float(request.form['reading_score']),
                writing_score=float(request.form['writing_score'])
            )
        except KeyError as e:
            flash(f'Missing data: {str(e)}', 'danger')
            return redirect(url_for('index'))
        except ValueError:
            flash('Invalid input. Please enter valid scores.', 'danger')
            return redirect(url_for('index'))

        pred_df = data.get_data_as_data_frame()
        logging.info(f"Data for prediction: {pred_df}")

        try:
            predict_pipeline = PredictPipeline()
            results = predict_pipeline.predict(pred_df)
            logging.info(f"Prediction results: {results}")
            return render_template('home.html', results=results[0])
        except Exception as e:
            logging.error(f"Error during prediction: {str(e)}")
            flash('Error during prediction. Please try again.', 'danger')
            return redirect(url_for('index'))

    return render_template('home.html')

if __name__ == "__main__":
    app.run(host="0.0.0.0", debug=True)
