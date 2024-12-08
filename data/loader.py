import pandas as pd
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.layers import TextVectorization
import numpy as np

class DataLoader():
    def __init__(self, train, valid, VOCAB_SIZE, SEQ_LENGTH, BATCH_SIZE):
        self.col = train.columns
        train = train[~train[self.col[0]].str.strip().eq('')]
        train = train[~train[self.col[1]].str.strip().eq('')]
        valid = valid[~valid[self.col[0]].str.strip().eq('')]
        valid = valid[~valid[self.col[1]].str.strip().eq('')]
        self.train, self.valid = train.copy(), valid.copy()
        self.SEQ_LENGTH = SEQ_LENGTH
        self.vectorizer = TextVectorization(
            max_tokens = VOCAB_SIZE,
            output_sequence_length = SEQ_LENGTH,
            output_mode = "int",
            standardize = self.custom_standardization
        )
        df = pd.concat([train, valid], ignore_index=True)
        self.train[self.col[1]] = self.train[self.col[1]].apply(lambda x: f"<start> {x} <end>")
        self.valid[self.col[1]] = self.valid[self.col[1]].apply(lambda x: f"<start> {x} <end>")
        text = df[self.col[0]].tolist() + df[self.col[1]].tolist()
        self.vectorizer.adapt(text)
        self.BATCH_SIZE = BATCH_SIZE

    def get_data(self):
        train = self.create_dataset(self.train)
        valid = self.create_dataset(self.valid)
        return train, valid, self.vectorizer
    
    def custom_standardization(self, input_string):
        return tf.strings.lower(input_string)
    
    def preprocess(self, text):
        text = self.vectorizer(text)
        text.set_shape([self.SEQ_LENGTH])
        return text

    def create_dataset(self, df):
        dataset = []
        for i in self.col:
            data = tf.data.Dataset.from_tensor_slices(df[i]).map(
                      self.preprocess, num_parallel_calls=tf.data.AUTOTUNE
                  )
            dataset.append(data)
        dataset = tf.data.Dataset.zip((dataset[0], dataset[1]))
        dataset = dataset.batch(self.BATCH_SIZE).shuffle(256).prefetch(tf.data.AUTOTUNE)
        return dataset
