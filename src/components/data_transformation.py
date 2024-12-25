import os
import pandas as pd
from src import logger
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split

import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

import keras
import tensorflow
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from src.entity.config_entity import (DataTransformationConfig)
from src.config.configuration import ConfigurationManager


class DataTransformation:
    def __init__(self, config: DataTransformationConfig):
        self.config = config

    def train_test_splitting(self):
        try:
            # Read the dataset
            df = pd.read_csv(self.config.data_path)

            # Drop duplicates
            df = df.drop_duplicates()
            logger.info(f"Shape after dropping duplicates: {df.shape}")

            # Encode the 'sentiment' column
            le = LabelEncoder()
            encoded_last_col = le.fit_transform(df.iloc[:, -1])
            df['sentiment'] = encoded_last_col

            # Tokenize the 'review' column
            tokenizer = Tokenizer(num_words=20000, oov_token="<OOV>")
            tokenizer.fit_on_texts(df['review'])
            sequences = tokenizer.texts_to_sequences(df['review'])

            # Pad the sequences
            df['review'] = list(pad_sequences(sequences, maxlen=100, padding='post'))


            # Split the data into training and test sets (75%, 25%)
            train, test = train_test_split(df, test_size=0.25, random_state=42)

            # Ensure the root directory exists
            os.makedirs(self.config.root_dir, exist_ok=True)

            # Save training and test sets to CSV
            train_path = os.path.join(self.config.root_dir, "train.csv")
            test_path = os.path.join(self.config.root_dir, "test.csv")
            train.to_csv(train_path, index=False)
            test.to_csv(test_path, index=False)

            # Log details
            logger.info("Split data into training and test sets")
            logger.info(f"Training set shape: {train.shape}")
            logger.info(f"Test set shape: {test.shape}")

            print(f"Training set shape: {train.shape}")
            print(f"Test set shape: {test.shape}")
        except Exception as e:
            logger.error(f"Error during train-test splitting: {e}")
            raise