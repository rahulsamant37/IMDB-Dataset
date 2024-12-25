from src.config.configuration import ConfigurationManager
from src.components.model_trainer import ModelTrainer
from src import logger
from pathlib import Path

STAGE_NAME="Model Trainer Stage"

class ModelTrainerPipeline:
    def __init__(self):
        pass

    def inititate_model_training(self):
        try:
            config = ConfigurationManager()
            model_types = ["rnn", "lstm", "gru"]

            for model_type in model_types:
                model_trainer_config = config.get_model_trainer_config(model_type=model_type)
                model_trainer = ModelTrainer(config=model_trainer_config)
                model_trainer.train(model_type=model_type)
        except Exception as e:
            raise e