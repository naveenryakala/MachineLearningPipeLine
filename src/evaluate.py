"""
Model Evaluation Script
=========================
This script loads a trained model and evaluates its performance.

Tasks:
- Load the trained model from disk
- Load test data
- Make predictions on the data
- Calculate accuracy and log metrics to MLflow
- Display performance results
"""

import pandas as pd
import pickle
from sklearn.metrics import accuracy_score
import yaml
import os
import mlflow
from dotenv import load_dotenv
from urllib.parse import urlparse

# Load environment variables (like MLflow credentials) from .env file
load_dotenv()

# Load parameters from params.yaml file
# Getting data and model paths from configuration
params = yaml.safe_load(open("params.yaml"))["train"]


def evaluate(data_path, model_path):
    """
    Evaluate the trained model on test data and log results.
    
    Args:
        data_path: Path to the data CSV file to evaluate
        model_path: Path to the trained model file (.pkl)
    """
    # Read the data from CSV file
    data = pd.read_csv(data_path)
    
    # Separate features (X) and target label (y)
    # Features: all columns except "Outcome"
    # Target: the "Outcome" column we want to predict
    X = data.drop(columns=["Outcome"])
    y = data["Outcome"]

    # Set MLflow tracking URI (where to log evaluation results)
    mlflow.set_tracking_uri("https://dagshub.com/ryakalanaveenreddy/machinelearningpipeline.mlflow")

    # Load the trained model from disk using pickle
    # The model was previously saved during training
    model = pickle.load(open(model_path, "rb"))

    # Make predictions on the data using the loaded model
    predictions = model.predict(X)
    
    # Calculate accuracy: percentage of correct predictions
    accuracy = accuracy_score(y, predictions)
    
    # Log the accuracy metric to MLflow for tracking
    mlflow.log_metric("accuracy", accuracy)
    
    # Print the accuracy score
    print(f"Model Accuracy Score: {accuracy}")


if __name__ == "__main__":
    # Run evaluation when this script is executed from command line
    # Use parameters from params.yaml file
    evaluate(params["data"], params["model"])


