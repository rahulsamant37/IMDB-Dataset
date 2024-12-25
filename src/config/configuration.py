from src.constants import *
from src.utils.common import read_yaml, create_directories
from src.entity.config_entity import (DataIngestionConfig,DataValidationConfig,DataTransformationConfig,ModelTrainerConfig,ModelEvalConfig)

class ConfigurationManager:
    def __init__(self,
                 config_filepath=CONFIG_FILE_PATH,
                 params_filepath = PARAMS_FILE_PATH,
                 schema_filepath = SCHEMA_FILE_PATH):
        self.config=read_yaml(config_filepath)
        self.params=read_yaml(params_filepath)
        self.schema=read_yaml(schema_filepath)

        create_directories([self.config.artifacts_root])


    def get_data_ingestion_config(self)-> DataIngestionConfig:
        config=self.config.data_ingestion
        create_directories([config.root_dir])

        data_ingestion_config=DataIngestionConfig(
            root_dir=config.root_dir,
            source_URL=config.source_URL,
            local_data_file=config.local_data_file,
            unzip_dir=config.unzip_dir

        )
        return data_ingestion_config
    
    def get_data_validation_config(self) -> DataValidationConfig:
        config = self.config.data_validation
        schema = self.schema.COLUMNS

        create_directories([config.root_dir])

        data_validation_config = DataValidationConfig(
            root_dir=config.root_dir,
            STATUS_FILE=config.STATUS_FILE,
            unzip_data_dir = config.unzip_data_dir,
            all_schema=schema,
        )

        return data_validation_config
    
    def get_data_transformation_config(self) -> DataTransformationConfig:
        config=self.config.data_transformation
        create_directories([config.root_dir])
        data_tranformation_config=DataTransformationConfig(
            root_dir=config.root_dir,
            data_path=config.data_path
        )
        return data_tranformation_config
    
    def get_model_trainer_config(self, model_type: str) -> ModelTrainerConfig:
        config=self.config.model_trainer
        params=self.params.model_params[model_type]
        schema=self.schema.TARGET_COLUMN
        create_directories([config.root_dir])
        model_trainer_config = ModelTrainerConfig(
            root_dir=config.root_dir,
            train_data_path=config.train_data_path,
            test_data_path=config.test_data_path,
            model_name=config.model_name,
            hidden_size=params["hidden_size"][0],  # Example: pick the first value
            num_layers=params["num_layers"][0],
            batch_size=params["batch_size"],
            learning_rate=params["learning_rate"][0],
            num_epochs=params["num_epochs"],
            dropout=params["dropout"][0],
            bidirectional=params["bidirectional"][0],
            embedding_dim=params["embedding_dim"][0],
            vocab_size=params["vocab_size"],
            max_length=params["max_length"],
            random_state=params["random_state"],
            target_column=schema['name']
        )
        return model_trainer_config
    
    def get_model_eval_config(self, model_type: str) -> ModelEvalConfig:

        config = self.config.model_evaluation
        params=self.params.model_params[model_type]
        schema = self.schema.TARGET_COLUMN

        create_directories([config.root_dir])

        model_eval_config = ModelEvalConfig(
            root_dir=config.root_dir,
            test_data_path=config.test_data_path,
            metric_file_name=config.metric_file_name,
            models=config.models,
            all_params=params,
            target_column=schema.name,
            mlflow_uri="https://dagshub.com/rahulsamantcoc2/IMDB-Dataset.mlflow",
            model_type=model_type
        )
        return model_eval_config