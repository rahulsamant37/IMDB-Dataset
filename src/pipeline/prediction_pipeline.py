import joblib
import numpy as np
from pathlib import Path


class PredictionPipeline:
    def __init__(self,model_name):
        if model_name is None:
            raise ValueError("Model name is required")
        elif model_name == 'rnn':
            self.model=joblib.load(Path('artifacts/model_trainer/model.joblib'))
        # Define a simple label mapping (assuming these were the class labels used during training)
        self.label_mapping = {0: 'setosa', 1: 'versicolor', 2: 'virginica'}
    
    def predict(self,data):
        prediction=self.model.predict(data)
        # Map the numeric label to the original class name
        decoded_prediction = self.label_mapping.get(prediction[0], "Unknown")
        return decoded_prediction