"""
Data Preprocessing Script
==========================
This script prepares raw data for machine learning model training.
It reads raw CSV data and saves it in a processed format.

Tasks:
- Load raw dataset from data/raw/ folder
- Clean and prepare data for training
- Save processed data to data/processed/ folder
"""

import pandas as pd
import sys
import yaml
import os

# Load configuration parameters from params.yaml file
# This keeps parameters separate from code for easy modification
params = yaml.safe_load(open("params.yaml"))["preprocess"]


def preprocess(input_path, output_path):
    """
    Preprocess raw data and save it for model training.
    
    Args:
        input_path (str): Path to raw input CSV file
        output_path (str): Path where processed CSV will be saved
    """
    # Read the raw CSV file into a pandas DataFrame
    data = pd.read_csv(input_path)

    # Create output directory if it doesn't exist
    # This ensures the directory is ready before saving the file
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    # Save the processed data to CSV file
    # header=None removes column names, index=False removes row numbers
    data.to_csv(output_path, header=None, index=False)
    print(f"Preprocessed data saved to {output_path}")


if __name__ == "__main__":
    # Run preprocessing with parameters from params.yaml
    # This ensures the script works when called from the command line
    preprocess(params["input"], params["output"])