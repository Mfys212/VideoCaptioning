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
        def apply_attention1(z):
            z_norm = self.layernorm[0](z)
            attention = self.attention1(query=z_norm, value=z_norm, key=z_norm, attention_mask=mask, training=training)
            return layers.Add()([z, attention])
        Z = tf.map_fn(apply_attention1, Z, dtype=tf.float32)
        Z_split = tf.split(Z, num_or_size_splits=self.num_patch, axis=2)
        def apply_attention2_ffn(z):
            z_norm = self.layernorm[1](z)
            attention = self.attention2(query=z_norm, value=z_norm, key=z_norm, attention_mask=mask, training=training)
            z = layers.Add()([z, attention])
            ffn = self.densel(self.dropout(self.dense(self.layernorm[2](z)), training=training))
            return layers.Add()([z, ffn])
        Z = tf.map_fn(apply_attention2_ffn, tf.stack(Z_split, axis=0), dtype=tf.float32)
        Z = tf.concat(tf.unstack(Z, axis=0), axis=1)
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
        Z = tf.map_fn(self.patch_embedding, Z, dtype=tf.float32)
        Z = tf.map_fn(lambda z: layers.Add()([z, self.Spositional_encoding(z)]), Z, dtype=tf.float32)
        for block in self.blocks:
            Z = block(Z, mask=mask, training=training)
            Z = tf.split(Z, num_or_size_splits=self.max_frames, axis=1)
        Z = tf.reshape(tf.concat(Z, axis=2), (tf.shape(inputs)[0], self.max_frames * self.spa_num_patch, self.d_models))
        return Z
