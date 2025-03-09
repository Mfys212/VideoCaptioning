import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from .module import *

class EncoderBlock(tf.keras.layers.Layer):
    def __init__(self, d_models, num_heads, **kwargs):
        super().__init__(**kwargs)
        d = d_models // num_heads
        self.attention1 = layers.MultiHeadAttention(num_heads=num_heads, key_dim=d, dropout=0.1)
        self.attention2 = layers.MultiHeadAttention(num_heads=num_heads, key_dim=d, dropout=0.1)
        self.layernorm1 = layers.LayerNormalization(epsilon=1e-6)
        self.layernorm2 = layers.LayerNormalization(epsilon=1e-6)
        self.layernorm3 = layers.LayerNormalization(epsilon=1e-6)
        self.dense1 = layers.Dense(d_models, activation="gelu")
        self.dense2 = layers.Dense(d_models)
        self.dropout = layers.Dropout(0.1)

    def call(self, Z, mask=None, training=True):
        Z = tf.stack(Z, axis=1)
        z_norm = self.layernorm1(Z)
        attn1 = self.attention1(query=z_norm, value=z_norm, key=z_norm, attention_mask=mask, training=training)
        Z = layers.Add()([Z, attn1])
        Z = tf.transpose(Z, perm=[0, 2, 1, 3])
        z_norm = self.layernorm2(Z)
        attn2 = self.attention2(query=z_norm, value=z_norm, key=z_norm, attention_mask=mask, training=training)
        Z = layers.Add()([Z, attn2])
        z_norm = self.layernorm3(Z)
        ffn_output = self.dense2(self.dropout(self.dense1(z_norm), training=training))
        Z = layers.Add()([Z, ffn_output])
        Z = tf.transpose(Z, perm=[0, 2, 1, 3])
        return tf.reshape(Z, [tf.shape(Z)[0], -1, tf.shape(Z)[-1]])

class Encoder(tf.keras.layers.Layer):
    def __init__(self, d_models, num_heads, num_l, max_frames, spatial_size, **kwargs):
        super().__init__(**kwargs)
        self.patch_embedding = PatchEmbedding(d_models, 1, 16, 16)
        self.spa_num_patch = (spatial_size // 16) ** 2
        self.Spositional_encoding = PositionalEncoding(sequence_length=self.spa_num_patch, embed_dim=d_models)
        self.Tpositional_encoding = PositionalEncoding(sequence_length=max_frames, embed_dim=d_models)
        self.blocks = [EncoderBlock(d_models, num_heads) for _ in range(num_l)]
        self.max_frames = max_frames
        self.d_models = d_models

    def call(self, inputs, training=True, mask=None):
        Z = tf.reshape(inputs, [-1, self.max_frames, self.spa_num_patch, self.d_models])
        Z = self.patch_embedding(Z)
        Z += self.Spositional_encoding(Z)
        for block in self.blocks:
            Z = block(Z, mask=mask, training=training)
        return tf.reshape(Z, [tf.shape(inputs)[0], -1, self.d_models])
