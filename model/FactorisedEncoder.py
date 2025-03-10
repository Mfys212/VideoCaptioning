import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from .module import *

class EncoderBlock(tf.keras.layers.Layer):
    def __init__(self, d_models, num_heads, **kwargs):
        super().__init__(**kwargs)
        d = d_models // num_heads
        self.attention = layers.MultiHeadAttention(num_heads=num_heads, key_dim=d, dropout=0.1)
        self.layernorm = [layers.LayerNormalization() for _ in range(2)]
        self.dense = layers.Dense(d_models, activation="gelu")
        self.densel = layers.Dense(d_models)
        self.dropout = layers.Dropout(0.1)

    def call(self, Z, mask=None, training=True):
        attention = self.attention(query=Z, value=Z, key=Z, attention_mask=mask, training=training)
        Z = self.layernorm[0](layers.Add()([Z, attention]))
        ffn = self.densel(self.dropout(self.dense(Z), training=training))
        Z = self.layernorm[1](layers.Add()([Z, ffn]))
        return Z

class Encoder(tf.keras.layers.Layer):
    def __init__(self, d_models, num_heads, num_l, max_frames, spatial_size, **kwargs):
        super().__init__(**kwargs)
        self.patch_embedding = PatchEmbedding(d_models, 1, 16, 16)
        num_patch = int((spatial_size/16)**2)
        self.Spositional_encoding = PositionalEncoding(sequence_length=num_patch, embed_dim=d_models)
        self.Tpositional_encoding = PositionalEncoding(sequence_length=max_frames, embed_dim=d_models)
        self.blocks_spatial = [EncoderBlock(d_models, num_heads) for _ in range(num_l)]
        self.blocks_temporal = [EncoderBlock(d_models, num_heads) for _ in range(num_l)]
        self.max_frames = max_frames

    def call(self, inputs, training=True, mask=None):
        Z = tf.split(inputs, num_or_size_splits=self.max_frames, axis=1)
        Z = [self.patch_embedding(z) for z in Z]
        Z = [layers.Add()([z, self.Spositional_encoding(z)]) for z in Z]
        Z_new = []
        for z in Z:
            for block in self.blocks_spatial:
                z = block(z, mask=mask, training=training)
            Z_new.append(z)
        Z = tf.stack(Z_new, axis=0)
        Z = tf.reduce_mean(Z, axis=0)
        Z = layers.Add()([Z, self.Tpositional_encoding(Z)])
        for block in self.blocks_temporal:
            Z = block(Z, mask=mask, training=training)
        return Z
