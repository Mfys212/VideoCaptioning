import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from .module import *

class EncoderBlock(tf.keras.layers.Layer):
    def __init__(self, d_models, num_heads, **kwargs):
        super().__init__(**kwargs)
        d = d_models // num_heads
        self.attention = MMultiHeadAttention(num_heads=num_heads, key_dim=d, d_models=d_models)
        self.layernorm = [layers.Normalization() for _ in range(2)]
        self.dense = layers.Dense(d_models, activation="gelu")
        self.densel = layers.Dense(d_models)
        self.dropout = layers.Dropout(0.1)

    def call(self, inputs, mask=None, training=True):
        Z, Zt, Zs = inputs
        attention = self.attention(query=Z, keys=Zs, keys2=Zt, values=Zs, values2=Zt, mask=mask, training=training)
        Z = self.layernorm[0](layers.Add()([Z, attention]))
        ffn = self.densel(self.dropout(self.dense(Z), training=training))
        Z = self.layernorm[1](layers.Add()([inputsf, ffn]))
        return Z

class Encoder(tf.keras.layers.Layer):
    def __init__(self, d_models, num_heads, num_l, max_frames, spatial_size, **kwargs):
        super().__init__(**kwargs)
        self.patch_embedding = PatchEmbedding2(d_models, 1, 16, 16)
        num_patch = int((max_frames*spatial_size**2) / (1*16*16))
        self.d_models = d_models
        self.nt = max_frames
        self.nh_nw = int((spatial_size/16)**2)
        self.positional_encoding = PositionalEncoding(sequence_length=num_patch, embed_dim=d_models)
        self.blocks = [EncoderBlock(d_models, num_heads) for _ in range(num_l)]
        self.flatten = layers.Reshape(target_shape=(-1, d_models))

    def call(self, inputs, training=True, mask=None):
        Z = self.flatten(inputs)
        Z = layers.Add()([Z, self.positional_encoding(Z)])
        inputs = tf.reshape(Z, tf.shape(inputs))
        batch_size = tf.shape(inputs)[0]
        Z, Zt, Zs = self.get_tem_spa(inputs, batch_size)
        for block in self.blocks:
            Z = block([Z, Zt, Zs], mask=mask, training=training)
            inputs = tf.reshape(Z, tf.shape(inputs))
            Z, Zt, Zs = self.get_tem_spa(inputs, batch_size)
        return Z

    def get_tem_spa(self, inputs, batch_size):
        Z = self.flatten(inputs)
        temporal_patches = tf.reshape(inputs, (batch_size, self.nt, self.nh_nw, self.d_models))
        Zt = tf.reduce_mean(temporal_patches, axis=2) 
        spatial_patches = tf.reshape(inputs, (batch_size, self.nt, self.nh_nw, self.d_models))
        Zs = tf.reduce_mean(spatial_patches, axis=1) 
        return Z, Zt, Zs
