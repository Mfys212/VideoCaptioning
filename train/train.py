import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.layers import TextVectorization
import numpy as np
from data.loader import *
from evaluation.evaluation import *
from model.Transformer import *

def TRANSFORMER(train, valid, batch=32, num_head=8, num_l=2, EMBED_DIM=512, SEQ_LENGTH=20, VOCAB_SIZE=10000):
    BATCH_SIZE = batch
    GetData = DataLoader(train, valid, VOCAB_SIZE, SEQ_LENGTH, BATCH_SIZE)
    train_dataset, valid_dataset, vectorization = GetData.get_data()
    encoder = TransformerEncoder(num_heads=num_head, num_l=num_l, EMBED_DIM=EMBED_DIM, vocab=vectorization.get_vocabulary())
    decoder = TransformerDecoder(num_heads=num_head, num_l=num_l, EMBED_DIM=EMBED_DIM, vocab=vectorization.get_vocabulary())
    model = MainModel(encoder, decoder, vectorization)

    def optim(optim=tf.keras.optimizers.Adam):
        loss = keras.losses.SparseCategoricalCrossentropy(
            from_logits=True, reduction="none"
        )

        class CustomSchedule(tf.keras.optimizers.schedules.LearningRateSchedule):
            def __init__(self, d_model, warmup_steps=4000):
                super(CustomSchedule, self).__init__()
                self.d_model = tf.cast(d_model, tf.float32)
                self.warmup_steps = warmup_steps

            def __call__(self, step):
                step = tf.cast(step, tf.float32)
                arg1 = tf.math.rsqrt(step)
                arg2 = step * (self.warmup_steps ** -1.5)
                return tf.math.rsqrt(self.d_model) * tf.math.minimum(arg1, arg2)

        lr_schedule = CustomSchedule(EMBED_DIM)

        optimizer = optim(
            learning_rate=lr_schedule,
            beta_1=0.9,
            beta_2=0.98,
            epsilon=1e-9
        )
        return optimizer, loss

    optimizer, loss = optim()
    model.compile(optimizer=optimizer, loss=loss)
    return model, train_dataset, valid_dataset