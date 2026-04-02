import pandas as pd
import pickle
from sklearn.metrics import accuracy_score
import yaml
import os
import mlflow
from dotenv import load_dotenv
from urllib.parse import urlparse

# Load environment variables from .env file
load_dotenv()


# Load parameters from param.yaml
params = yaml.safe_load(open("params.yaml"))["train"]

def evaluate(data_path,model_path):
    data=pd.read_csv(data_path)
    X=data.drop(columns=["Outcome"])
    y=data["Outcome"]

    mlflow.set_tracking_uri("https://dagshub.com/ryakalanaveenreddy/machinelearningpipeline.mlflow")

    ## loading the model from the disk
    model = pickle.load(open(model_path, "rb"))

    predictions=model.predict(X)
    accuracy=accuracy_score(y,predictions)
    
    ## log metrics to MLflow
    mlflow.log_metric("accuracy", accuracy)
    print(f"Model Accuracy Score: {accuracy}")

if __name__ == "__main__":
    evaluate(params["data"],params["model"])


