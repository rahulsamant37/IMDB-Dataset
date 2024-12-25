import os
import pandas as pd
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix
from urllib.parse import urlparse
import mlflow
import mlflow.keras
import numpy as np
import joblib
from sklearn.preprocessing import LabelEncoder
from src import logger
from typing import List, Union
from pathlib import Path
from src.utils.common import save_json
from src.config.configuration import ModelEvalConfig

import os
os.environ['MLFLOW_TRACKING_URI']="https://dagshub.com/rahulsamantcoc2/IMDB-Dataset.mlflow"
os.environ['MLFLOW_TRACKING_USERNAME']="rahulsamantcoc2"
os.environ['MLFLOW_TRACKING_PASSWORD']="33607bcb15d4e7a7cca29f0f443d16762cc15549"

class ModelEval:
    def __init__(self, config: ModelEvalConfig):
        self.config = config
        self.vocab_size = 10000
        self.best_model_metrics = None
        self.best_model_type = None

    def preprocess_features(self, data: pd.DataFrame) -> np.ndarray:
        features_list = []
        for _, row in data.iterrows():
            feature_str = row.iloc[0]
            if isinstance(feature_str, str):
                features = np.array([
                    min(int(x), self.vocab_size - 1) 
                    for x in feature_str.strip('[]').split()
                ])
            else:
                features = np.minimum(np.array(feature_str), self.vocab_size - 1)
            features_list.append(features)
        return np.vstack(features_list)

    def Eval(self, actual, pred):
        pred_binary = (pred >= 0.5).astype(int)
        actual_binary = actual.astype(int)
        
        accuracy = accuracy_score(actual_binary, pred_binary)
        precision = precision_score(actual_binary, pred_binary)
        recall = recall_score(actual_binary, pred_binary)
        f1 = f1_score(actual_binary, pred_binary)
        auc = roc_auc_score(actual_binary, pred)
        cm = confusion_matrix(actual_binary, pred_binary)
        return accuracy, precision, recall, f1, auc, cm

    def update_best_model(self, metrics, model_type):
        accuracy, _, _, f1, auc, _ = metrics
        current_score = (accuracy + f1 + auc) / 3  # Combined metric
        
        if self.best_model_metrics is None or current_score > self.best_model_metrics['combined_score']:
            self.best_model_metrics = {
                'model_type': model_type,
                'accuracy': accuracy,
                'f1': f1,
                'auc': auc,
                'combined_score': current_score
            }
            self.best_model_type = model_type

    def log_into_mlflow(self):
        test_data = pd.read_csv(self.config.test_data_path)
        model_path = self.config.models.get(self.config.model_type, {}).get('model_path')
        
        if model_path is None:
            raise ValueError(f"Model path not found for model type: {self.config.model_type}")
            
        model = joblib.load(model_path)
        test_x = self.preprocess_features(test_data.drop([self.config.target_column], axis=1))
        test_y = test_data[self.config.target_column].values

        mlflow.set_registry_uri(self.config.mlflow_uri)
        tracking_url_type_store = urlparse(mlflow.get_tracking_uri()).scheme
        
        with mlflow.start_run():
            predicted_qualities = model.predict(test_x)
            metrics = self.Eval(test_y, predicted_qualities)
            self.update_best_model(metrics, self.config.model_type)
            
            accuracy, precision, recall, f1, auc, cm = metrics
            scores = {
                "accuracy": float(accuracy),
                "precision": float(precision),
                "recall": float(recall),
                "f1": float(f1),
                "auc": float(auc),
                "confusion_matrix": cm.tolist()
            }
            
            save_json(path=Path(self.config.metric_file_name), data=scores)
            mlflow.log_params(self.config.all_params)
            
            for metric_name, value in scores.items():
                if metric_name != "confusion_matrix":
                    mlflow.log_metric(metric_name, value)
            
            if tracking_url_type_store != "file":
                mlflow.sklearn.log_model(model, "model", registered_model_name="IMDB")
            else:
                mlflow.sklearn.log_model(model, "model")