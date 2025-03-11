import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from .module import *

class EncoderBlock(tf.keras.layers.Layer):
    def __init__(self, d_models, num_heads, num_patch, **kwargs):
        super().__init__(**kwargs)
        d = d_models // num_heads
        self.attention1 = layers.MultiHeadAttention(num_heads=num_heads, key_dim=d, dropout=0.1)
        self.attention2 = layers.MultiHeadAttention(num_heads=num_heads, key_dim=d, dropout=0.1)
        self.layernorm = [layers.LayerNormalization() for _ in range(3)]
        self.dense = layers.Dense(d_models, activation="gelu")
        self.densel = layers.Dense(d_models)
        self.dropout = layers.Dropout(0.1)
        self.num_patch = num_patch

    def call(self, Z, mask=None, training=True):
        Z_new = []
        for z in Z:
            z_norm = self.layernorm[0](z)
            attention = self.attention1(query=z_norm, value=z_norm, key=z_norm, attention_mask=mask, training=training)
            Z_new.append(layers.Add()([z, attention]))
        Z = tf.stack(Z_new, axis=1)
        Z_split = tf.split(Z, num_or_size_splits=self.num_patch, axis=2)
        Z_new = []
        for z in Z_split:
            z_norm = self.layernorm[1](z)
            attention = self.attention2(query=z_norm, value=z_norm, key=z_norm, attention_mask=mask, training=training)
            z = layers.Add()([z, attention])
            ffn = self.densel(self.dropout(self.dense(self.layernorm[2](z)), training=training))
            Z_new.append(layers.Add()([z, ffn]))
        Z = tf.concat(Z_new, axis=1)
        return Z

class Encoder(tf.keras.models.Model):
    def __init__(self, d_models, num_heads, num_l, max_frames, spatial_size, **kwargs):
        super().__init__(**kwargs)
        self.patch_embedding = PatchEmbedding(d_models, 1, 16, 16)
        self.spa_num_patch = int((spatial_size / 16) ** 2)
        self.Spositional_encoding = PositionalEncoding(sequence_length=self.spa_num_patch, embed_dim=d_models)
        self.blocks = [EncoderBlock(d_models, num_heads, self.spa_num_patch) for _ in range(num_l)]
        self.max_frames = max_frames
        self.d_models = d_models

    def call(self, inputs, training=True, mask=None):
        Z = tf.split(inputs, num_or_size_splits=self.max_frames, axis=1)
        Z = [self.patch_embedding(z) for z in Z]
        Z = [layers.Add()([z, self.Spositional_encoding(z)]) for z in Z]
        for block in self.blocks:
            Z = block(Z, mask=mask, training=training)
            Z = tf.split(Z, num_or_size_splits=self.max_frames, axis=1)
        Z = tf.reshape(tf.concat(Z, axis=2), (tf.shape(inputs)[0], self.max_frames * self.spa_num_patch, self.d_models))
        return Z
