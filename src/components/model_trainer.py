import pandas as pd
import os
from src import logger
from pathlib import Path
import keras
import tensorflow
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, Dense, Dropout, SimpleRNN, LSTM, GRU
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.layers import Bidirectional
import joblib

from src.utils.common import read_yaml
from src.config.configuration import ModelTrainerConfig
from src.constants import SCHEMA_FILE_PATH
from typing import Any
from tensorflow.keras.models import Model

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

def save_bin(data: Any, path: Path):
    """Save binary file

    Args:
        data (Any): Data to be saved as binary
        path (Path): Path to binary file
    """
    if not isinstance(data, (Model, object)):  # Adjust types if needed
        raise TypeError(f"Invalid data type: {type(data)} for save_bin.")
    joblib.dump(value=data, filename=path)
    logger.info(f"Binary file saved at: {path}")

class ModelTrainer:
    def __init__(self, config: ModelTrainerConfig):
        self.config = config
        self.schema = read_yaml(SCHEMA_FILE_PATH)

    def train(self, model_type: str):
        # Load data
        train_df = pd.read_csv(self.config.train_data_path)
        test_df = pd.read_csv(self.config.test_data_path)
    
        # Tokenization
        tokenizer = Tokenizer(num_words=self.config.vocab_size)
        tokenizer.fit_on_texts(train_df['review'])
    
        # Convert text to sequences
        X_train = tokenizer.texts_to_sequences(train_df['review'])
        X_test = tokenizer.texts_to_sequences(test_df['review'])
    
        # Pad sequences
        X_train = pad_sequences(X_train, maxlen=self.config.max_length)
        X_test = pad_sequences(X_test, maxlen=self.config.max_length)
    
        # Get labels
        y_train = train_df[self.config.target_column]
        y_test = test_df[self.config.target_column]
    
        # Build model
        model = Sequential()
        embedding_dim = self.config.embedding_dim
        model.add(Embedding(self.config.vocab_size, embedding_dim, input_length=self.config.max_length))
    
        num_layers = self.config.num_layers
        hidden_size = self.config.hidden_size
        dropout = self.config.dropout
        bidirectional = self.config.bidirectional
    
        # Add RNN layers
        for i in range(num_layers):
            return_sequences = i < num_layers - 1
            if 'lstm' in self.config.model_name:
                layer = LSTM(hidden_size, return_sequences=return_sequences)
            elif 'gru' in self.config.model_name:
                layer = GRU(hidden_size, return_sequences=return_sequences)
            else:
                layer = SimpleRNN(hidden_size, return_sequences=return_sequences)
    
            if bidirectional:
                layer = Bidirectional(layer)
    
            model.add(layer)
            model.add(Dropout(dropout))
    
        # Add output layer
        model.add(Dense(1, activation='sigmoid'))
    
        # Compile model
        learning_rate = self.config.learning_rate
        model.compile(
            optimizer=Adam(learning_rate=learning_rate),
            loss='binary_crossentropy',
            metrics=['accuracy']
        )
    
        # Train model
        history = model.fit(
            X_train, y_train,
            batch_size=self.config.batch_size,
            epochs=self.config.num_epochs,
            validation_data=(X_test, y_test),
            verbose=1
        )
    
        # Create directories for saving
        model_dir = os.path.join(self.config.root_dir, model_type)
        os.makedirs(model_dir, exist_ok=True)
    
        # Save model and tokenizer
        model_path = os.path.join(model_dir, f"{self.config.model_name}.joblib")
        tokenizer_path = os.path.join(model_dir, f"{self.config.model_name}_tokenizer.joblib")
    
        save_bin(model, model_path)
        save_bin(tokenizer, tokenizer_path)
    
        return model, history, tokenizer
    