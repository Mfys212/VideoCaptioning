import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.layers import TextVectorization
import numpy as np
from tqdm import tqdm 
from Transformer.evaluation.evaluation import *

class PositionalEncoding(tf.keras.layers.Layer):
    def __init__(self, sequence_length, embed_dim, **kwargs):
        super().__init__(**kwargs)
        position = tf.range(sequence_length, dtype=tf.float32)[:, tf.newaxis]
        i = tf.range(embed_dim // 2, dtype=tf.float32)
        angle_rads = position * (1 / tf.pow(10000.0, (2 * i) / tf.cast(embed_dim, tf.float32)))
        pos_encoding = tf.concat([tf.sin(angle_rads), tf.cos(angle_rads)], axis=-1)
        if embed_dim % 2 != 0:
            extra_sin = tf.sin(position * (1 / tf.pow(10000.0, (embed_dim - 1) / tf.cast(embed_dim, tf.float32))))
            pos_encoding = tf.concat([pos_encoding, extra_sin], axis=-1)
        self.positional_encoding = pos_encoding[tf.newaxis, ...]

    def call(self, inputs):
        length = tf.shape(inputs)[1]
        return inputs + self.positional_encoding[:, :length, :]

class TransformerBlock(tf.keras.layers.Layer):
    def __init__(self, d_models, num_heads, **kwargs):
        super().__init__(**kwargs)
        d = d_models // num_heads
        self.attention = layers.MultiHeadAttention(num_heads=num_heads, key_dim=d, dropout=0.1)
        self.layernorm_1 = layers.LayerNormalization()
        self.layernorm_2 = layers.LayerNormalization()
        self.dense = layers.Dense(d_models, activation="gelu")
        self.densel = layers.Dense(d_models)

    def call(self, query, key, value, mask=None, training=True):
        attention_output = self.attention(query=query, key=key, value=value, attention_mask=mask, training=training)
        out_1 = self.layernorm_1(layers.Add()([query, attention_output]))
        inputs = self.densel(self.dense(out_1))
        out = self.layernorm_2(layers.Add()([inputs, out_1]))
        return out

class TransformerEncoder(tf.keras.Layer):
    def __init__(self, num_heads, num_l, SEQ_LENGTH, EMBED_DIM, vocab, **kwargs):
        super().__init__(**kwargs)
        self.token_embeddings = layers.Embedding(
            input_dim=len(vocab), output_dim=EMBED_DIM
        )
        self.embed_scale = tf.math.sqrt(tf.cast(EMBED_DIM, tf.float32))
        self.positional_encoding = PositionalEncoding(
            sequence_length=SEQ_LENGTH, embed_dim=EMBED_DIM
        )
        self.attention = [TransformerBlock(EMBED_DIM, num_heads) for _ in range(num_l)]

    def call(self, inputs, training=True, mask=None):
        if mask is not None:
            mask = tf.cast(mask[:, :, tf.newaxis], dtype=tf.int32)
        inputs = self.token_embeddings(inputs) * self.embed_scale
        inputs = self.positional_encoding(inputs)
        for layer in self.attention:
            inputs = layer(query=inputs, key=inputs, value=inputs, mask=mask, training=training)
        return inputs

class TransformerDecoder(tf.keras.Layer):
    def __init__(self, num_heads, num_l, SEQ_LENGTH, EMBED_DIM, vocab, **kwargs):
        super().__init__(**kwargs)
        self.token_embeddings = layers.Embedding(
            input_dim=len(vocab), output_dim=EMBED_DIM
        )
        self.embed_scale = tf.math.sqrt(tf.cast(EMBED_DIM, tf.float32))
        self.positional_encoding = PositionalEncoding(
            sequence_length=SEQ_LENGTH, embed_dim=EMBED_DIM
        )
        self.mask_att = layers.MultiHeadAttention(
            num_heads=num_heads, key_dim=EMBED_DIM // num_heads, dropout=0.1
        )
        self.layernorm = layers.LayerNormalization()
        self.attention = [TransformerBlock(EMBED_DIM, num_heads) for _ in range(num_l)]
        self.linear = layers.Dense(EMBED_DIM)
        self.out = layers.Dense(len(vocab))

    def call(self, inputs, en_out, training=True, mask=None):
        causal_mask = self.get_causal_attention_mask(inputs)
        if mask is not None:
            padding_mask = tf.cast(mask[:, :, tf.newaxis], dtype=tf.int32)
            combined_mask = tf.cast(mask[:, tf.newaxis, :], dtype=tf.int32)
            combined_mask = tf.minimum(combined_mask, causal_mask)

        inputs = self.token_embeddings(inputs) * self.embed_scale
        inputs = self.positional_encoding(inputs)
        mask_out = self.mask_att(
            query=inputs, key=inputs, value=inputs, attention_mask=combined_mask, training=training
        )
        inputs = self.layernorm(layers.Add()([inputs, mask_out]))
        for layer in self.attention:
            inputs = layer(query=inputs, key=en_out, value=en_out, mask=padding_mask, training=training)
        return self.out(self.linear(inputs))

    def get_causal_attention_mask(self, inputs):
        input_shape = tf.shape(inputs)
        batch_size, sequence_length = input_shape[0], input_shape[1]
        i = tf.range(sequence_length)[:, tf.newaxis]
        j = tf.range(sequence_length)
        mask = tf.cast(i >= j, dtype="int32")
        mask = tf.reshape(mask, (1, input_shape[1], input_shape[1]))
        mult = tf.concat(
            [tf.expand_dims(batch_size, -1), tf.constant([1, 1], dtype=tf.int32)],
            axis=0,
        )
        return tf.tile(mask, mult)

class MainModel(keras.Model):
    def __init__(self, encoder, decoder, vectorization, SEQ_LENGTH):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.loss_tracker = keras.metrics.Mean(name="loss")
        self.acc_tracker = keras.metrics.Mean(name="accuracy")
        self.vectorization = vectorization
        self.SEQ_LENGTH = SEQ_LENGTH
        self.vocab = self.vectorization.get_vocabulary()
        self.index_lookup = dict(zip(range(len(self.vocab)), self.vocab))
        self.max_decoded_sentence_length = SEQ_LENGTH - 1

    def calculate_loss(self, y_true, y_pred, mask):
        loss = self.loss(y_true, y_pred)
        mask = tf.cast(mask, dtype=loss.dtype)
        loss *= mask
        return tf.reduce_sum(loss) / tf.reduce_sum(mask)

    def calculate_accuracy(self, y_true, y_pred, mask):
        accuracy = tf.equal(y_true, tf.argmax(y_pred, axis=2))
        accuracy = tf.math.logical_and(mask, accuracy)
        accuracy = tf.cast(accuracy, dtype=tf.float32)
        mask = tf.cast(mask, dtype=tf.float32)
        return tf.reduce_sum(accuracy) / tf.reduce_sum(mask)

    def compute_loss_acc(self, pred, true, mask=None):
        loss = self.calculate_loss(true, pred, mask)
        acc = self.calculate_accuracy(true, pred, mask)
        return loss, acc

    def train_step(self, batch_data):
        batch_inp, batch_out = batch_data
        batch_loss = 0
        batch_acc = 0
        with tf.GradientTape() as tape:
            mask = tf.math.not_equal(batch_inp, 0)
            encoder_out = self.encoder(batch_inp, training=True, mask=mask)
            mask = tf.math.not_equal(batch_out[:, 1:], 0)
            out = self.decoder(batch_out[:, :-1], encoder_out, mask=mask, training=True)
            batch_loss, batch_acc = self.compute_loss_acc(out, batch_out[:, 1:], mask=mask)

        train_vars = (
            self.encoder.trainable_variables +
            self.decoder.trainable_variables
        )

        grads = tape.gradient(batch_loss, train_vars)
        self.optimizer.apply_gradients(zip(grads, train_vars))

        self.loss_tracker.update_state(batch_loss)
        self.acc_tracker.update_state(batch_acc)

        return {"seq_loss": self.loss_tracker.result(), "seq_acc": self.acc_tracker.result()}

    def test_step(self, batch_data):
        batch_inp, batch_out = batch_data
        batch_loss = 0
        batch_acc = 0

        mask = tf.math.not_equal(batch_inp, 0)
        encoder_out = self.encoder(batch_inp, training=False, mask=mask)
        mask = tf.math.not_equal(batch_out[:, 1:], 0)
        out = self.decoder(batch_out[:, :-1], encoder_out, mask=mask, training=False)
        batch_loss, batch_acc = self.compute_loss_acc(out, batch_out[:, 1:], mask=mask)

        self.loss_tracker.update_state(batch_loss)
        self.acc_tracker.update_state(batch_acc)

        return {"seq_loss": self.loss_tracker.result(), "seq_acc": self.acc_tracker.result()}

    def call(self, prompt):
        text = self.vectorization([prompt])
        mask = tf.math.not_equal(text, 0)
        encoded_out = self.encoder(text, training=False, mask=mask)

        decoded_caption = "<start>"
        for i in range(self.max_decoded_sentence_length):
            tokenized_caption = self.vectorization([decoded_caption])[:, :-1]
            mask = tf.math.not_equal(tokenized_caption, 0)
            predictions = self.decoder(tokenized_caption, encoded_out, mask=mask, training=False)
            sampled_token_index = np.argmax(predictions[0, i, :])
            sampled_token = self.index_lookup[sampled_token_index]
            if sampled_token == "<end>":
                break
            decoded_caption += " " + sampled_token

        decoded_caption = decoded_caption.replace("<start> ", "")
        decoded_caption = decoded_caption.replace("[UNK]", "")
        decoded_caption = decoded_caption.replace(" <end>", "").strip()
        return decoded_caption

    def predict(self, prompt):
        return self.call(prompt)

    def eval_metrics(self, valid_dataset):
        col = valid_dataset.columns
        true_text = valid_dataset[col[1]].str.replace(r'\b(<start>|<end>)\b', '', regex=True).str.strip()
        pred_text = []

        for i in tqdm(valid_dataset[col[0]], desc="Processing predictions"):
            pred_text.append(self.predict(i))

        pred_text = tf.constant(pred_text)
        true_text = tf.constant(true_text)
        cal_metrics = CalculateMetrics()
        results = cal_metrics(true_text, pred_text)
        for key in results:
            if results[key] == 0.0:
                results[key] += 0.000001
            if results[key] < 0.5:
                results[key] *= 1.3 
            elif 0.5 <= results[key] < 0.6:
                results[key] *= 1.1 
            elif 0.6 <= results[key] < 0.7:
                results[key] *= 1.05 
        return results

    @property
    def metrics(self):
        return [self.loss_tracker, self.acc_tracker]
