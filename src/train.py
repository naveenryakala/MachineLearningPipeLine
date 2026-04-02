"""
Model Training Script
=======================
This script trains a Random Forest Classifier on preprocessed data.

Key Functions:
- Load preprocessed data
- Split data into training and testing sets
- Find best hyperparameters using Grid Search
- Train the best model
- Log metrics and results to MLflow for experiment tracking
- Save the trained model to disk
"""

import pandas as pd
from sklearn.ensemble import RandomForestClassifier
import pickle
import yaml
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from mlflow.models import infer_signature
import os
from dotenv import load_dotenv

from sklearn.model_selection import train_test_split, GridSearchCV
from urllib.parse import urlparse

import mlflow

# Load environment variables (like MLflow credentials) from .env file
load_dotenv()

# Load parameters from params.yaml file
# Parameters are stored separately so we can change them without changing code
params = yaml.safe_load(open("params.yaml"))["train"]


def hyperparameter_tuning(X_train, y_train, param_grid):
    """
    Find the best hyperparameters for Random Forest Classifier.
    
    Hyperparameters are settings that control how the model learns.
    GridSearchCV tests different combinations to find the best ones.
    
    Args:
        X_train: Training features (input data)
        y_train: Training labels (output data)
        param_grid: Dictionary of hyperparameters to test
        
    Returns:
        GridSearchCV object with the best model found
    """
    # Create a Random Forest Classifier
    rf = RandomForestClassifier()
    
    # Grid Search tests all combinations of hyperparameters
    # cv=5: Use 5-fold cross-validation for more reliable results
    # n_jobs=-1: Use all available CPU cores for faster execution
    grid_search = GridSearchCV(estimator=rf, param_grid=param_grid, cv=5, n_jobs=-1, verbose=2)
    
    # Train the model with different hyperparameter combinations
    grid_search.fit(X_train, y_train)
    
    return grid_search


def train(data_path, model_path, random_state, n_estimators, max_depth):
    """
    Train a Random Forest model and log results to MLflow.
    
    Args:
        data_path: Path to the training data CSV file
        model_path: Path where the trained model will be saved
        random_state: Seed for reproducibility
        n_estimators: Number of trees in the forest
        max_depth: Maximum depth of each tree
    """
    # Read the preprocessed data from CSV file
    data = pd.read_csv(data_path)
    
    # Separate features (X) and target label (y)
    # Features: all columns except "Outcome"
    # Target: the "Outcome" column we want to predict
    X = data.drop(columns=["Outcome"])
    y = data["Outcome"]

    # Set MLflow tracking URI (where to log experiments)
    mlflow.set_tracking_uri("https://dagshub.com/ryakalanaveenreddy/machinelearningpipeline.mlflow")

    # Start an MLflow experiment run (session to track this training)
    with mlflow.start_run():
        # Split data: 80% for training, 20% for testing
        # test_size=0.20 means 20% of data goes to testing
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20)
        
        # Infer model signature (what inputs/outputs the model expects)
        signature = infer_signature(X_train, y_train)

        # Define a grid of hyperparameters to test
        # The model will try all combinations of these values
        param_grid = {
            "n_estimators": [100, 200],           # Number of trees to create
            "max_depth": [5, 10, None],           # Maximum tree depth
            "min_samples_split": [2, 5],          # Minimum samples to split a node
            "min_samples_leaf": [1, 2]            # Minimum samples in a leaf node
        }

        # Run hyperparameter tuning to find the best combination
        grid_search = hyperparameter_tuning(X_train, y_train, param_grid)

        # Get the best model from the grid search
        best_model = grid_search.best_estimator_

        # Make predictions on test data using the best model
        y_pred = best_model.predict(X_test)
        
        # Calculate accuracy: how many predictions were correct?
        accuracy = accuracy_score(y_test, y_pred)
        print(f"Accuracy Score: {accuracy}")

        # Log the accuracy metric to MLflow for tracking
        mlflow.log_metric("accuracy_score", accuracy)
        
        # Log the best hyperparameters found
        mlflow.log_param("best_n_estimators", grid_search.best_params_["n_estimators"])
        mlflow.log_param("best_max_depth", grid_search.best_params_["max_depth"])
        mlflow.log_param("best_samples_split", grid_search.best_params_["min_samples_split"])
        mlflow.log_param("best_samples_leaf", grid_search.best_params_["min_samples_leaf"])

        # Create confusion matrix and classification report
        # These show detailed performance metrics
        cm = confusion_matrix(y_test, y_pred)
        cr = classification_report(y_test, y_pred)

        # Log the confusion matrix and report to MLflow as text files
        mlflow.log_text(str(cm), "confusion_matrix.txt")
        mlflow.log_text(cr, "classification_report.txt")

        # Check the type of tracking URI (local file vs remote server)
        tracking_url_type_store = urlparse(mlflow.get_tracking_uri()).scheme

        # If using a remote MLflow server, register the model
        if tracking_url_type_store != "file":
            mlflow.sklearn.log_model(best_model, "model", registered_model_name="Best Model")
        # Otherwise, just log it locally
        else:
            mlflow.sklearn.log_model(best_model, "model", signature=signature)

        # Create the directory to save the model if it doesn't exist
        os.makedirs(os.path.dirname(model_path), exist_ok=True)

        # Save the best model to disk as a pickle file (.pkl)
        # This allows us to load and use it later without retraining
        filename = model_path
        pickle.dump(best_model, open(filename, "wb"))

        print(f"Model saved to {model_path}")


if __name__ == "__main__":
    # Run training when this script is executed from command line
    # Use parameters from params.yaml file
    train(params["data"], params["model"], params["random_state"], params["n_estimators"], params["max_depth"])


