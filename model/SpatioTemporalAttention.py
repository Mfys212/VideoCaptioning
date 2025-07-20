import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from .module import *
try:
    from VideoCaptioning import set_seed
    set_seed()
except:
    pass

class EncoderBlock(tf.keras.layers.Layer):
    def __init__(self, d_models, num_heads, **kwargs):
        super().__init__(**kwargs)
        d = d_models // num_heads
        self.attention = layers.MultiHeadAttention(num_heads=num_heads, key_dim=d, dropout=0.1)
        self.layernorm = [layers.LayerNormalization() for _ in range(3)]
        self.dense = layers.Dense(d_models, activation="gelu")
        self.densel = layers.Dense(d_models)
        self.dropout = layers.Dropout(0.1)

    def call(self, Z, mask=None, training=True):
        Z_norm = self.layernorm[0](Z)
        attention = self.attention(query=Z_norm, value=Z_norm, key=Z_norm, attention_mask=mask, training=training)
        Z = self.layernorm[1](layers.Add()([Z, attention]))
        ffn = self.densel(self.dropout(self.dense(Z), training=training))
        Z = self.layernorm[2](layers.Add()([Z, ffn]))
        return Z

class Encoder(tf.keras.models.Model):
    def __init__(self, d_models, num_heads, num_l, max_frames, spatial_size, **kwargs):
        super().__init__(**kwargs)
        self.patch_embedding = PatchEmbedding(d_models, 2, 16, 16)
        num_patch = int((max_frames*spatial_size**2) / (2*16*16))
        self.positional_encoding = PositionalEncoding(sequence_length=num_patch, embed_dim=d_models)
        self.blocks = [EncoderBlock(d_models, num_heads) for _ in range(num_l)]

    def call(self, inputs, training=True, mask=None):
        Z = self.patch_embedding(inputs)
        Z = layers.Add()([Z, self.positional_encoding(Z)])
        for block in self.blocks:
            Z = block(Z, mask=mask, training=training)
        return Z
