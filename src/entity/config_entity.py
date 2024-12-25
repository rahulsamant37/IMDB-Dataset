from dataclasses import dataclass
from pathlib import Path
from typing import List, Union, Dict

@dataclass
class DataIngestionConfig:
    root_dir: Path
    source_URL: str
    local_data_file: Path
    unzip_dir: Path

@dataclass
class DataValidationConfig:
    root_dir: Path
    STATUS_FILE: str
    unzip_data_dir: Path
    all_schema: dict

@dataclass
class DataTransformationConfig:
    root_dir: Path
    data_path: Path

@dataclass
class ModelTrainerConfig:
    root_dir: Path
    train_data_path: Path
    test_data_path: Path
    model_name: str
    hidden_size: Union[int, List[int]]
    num_layers: Union[int, List[int]]
    batch_size: int
    learning_rate: Union[float, List[float]]
    num_epochs: int
    dropout: Union[float, List[float]]
    bidirectional: Union[bool, List[bool]]
    embedding_dim: List[int]
    vocab_size: int
    max_length: int
    random_state: int
    target_column: str

@dataclass
class ModelEvalConfig:
    root_dir: str
    test_data_path: str
    metric_file_name: str
    models: Dict[str, Dict[str, str]]
    all_params: dict
    target_column: str
    mlflow_uri: str
    model_type: str