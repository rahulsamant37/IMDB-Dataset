from src.config.configuration import ConfigurationManager
from src.components.model_evaluation import ModelEval
from src import logger
from pathlib import Path

STAGE_NAME="Model Trainer Stage"

class ModelEvaluationPipeline:
    def __init__(self):
        pass

    def inititate_model_eval(self):
        best_model_metrics = None
        model_evals = []

        try:
            config = ConfigurationManager()

            for model_type in ['rnn', 'lstm', 'gru']:
                model_eval_config = config.get_model_eval_config(model_type=model_type)
                model_eval = ModelEval(config=model_eval_config)
                model_eval.log_into_mlflow()
                model_evals.append(model_eval)

            # Find best model
            best_model = max(model_evals, key=lambda x: x.best_model_metrics['combined_score'])
            print(f"\nBest Model: {best_model.best_model_type}")
            print(f"Metrics: Accuracy={best_model.best_model_metrics['accuracy']:.4f}, "
                  f"F1={best_model.best_model_metrics['f1']:.4f}, "
                  f"AUC={best_model.best_model_metrics['auc']:.4f}")

        except Exception as e:
            raise e