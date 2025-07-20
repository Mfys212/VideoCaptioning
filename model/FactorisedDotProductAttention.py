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
    def __init__(self, d_models, num_heads, max_frames, num_p_spa, dot_1=True, **kwargs):
        super().__init__(**kwargs)
        d = d_models // num_heads
        self.dot_1 = dot_1
        if dot_1:
            self.attention = MMultiHeadAttention(num_heads=num_heads, key_dim=d, d_models=d_models, nt=max_frames, nh_nw=num_p_spa)
        else:
            self.attention1 = layers.MultiHeadAttention(num_heads=num_heads//2, key_dim=d, dropout=0.1, output_shape=d_models//2)
            self.attention2 = layers.MultiHeadAttention(num_heads=num_heads//2, key_dim=d, dropout=0.1, output_shape=d_models//2)
        self.layernorm = [layers.LayerNormalization() for _ in range(2)]
        self.dense = layers.Dense(d_models, activation="gelu")
        self.densel = layers.Dense(d_models)
        self.dropout = layers.Dropout(0.1)
        self.nt = max_frames
        self.nh_nw = num_p_spa
        self.d_model = d_models

    def call(self, Z, mask=None, training=True):
        batch_size = tf.shape(Z)[0]
        st = tf.reshape(Z, (batch_size, self.nt, self.nh_nw, self.d_model))
        t = tf.reduce_mean(st, axis=2)
        s = tf.reduce_mean(st, axis=1)
        if self.dot_1:
            attention = self.attention1(query=Z, keys=Z, values=Z, mask=mask, training=training)
        else:
            attention1 = self.attention1(query=Z, value=s, key=s, attention_mask=mask, training=training)
            attention2 = self.attention2(query=Z, value=t, key=t, attention_mask=mask, training=training)
            attention = tf.concat([attention1, attention2], axis=-1)
        Z = self.layernorm[0](layers.Add()([Z, attention]))
        ffn = self.densel(self.dropout(self.dense(Z), training=training))
        Z = self.layernorm[1](layers.Add()([Z, ffn]))
        return Z

class Encoder(tf.keras.models.Model):
    def __init__(self, d_models, num_heads, num_l, max_frames, spatial_size, **kwargs):
        super().__init__(**kwargs)
        self.patch_embedding = PatchEmbedding(d_models, 2, 16, 16)
        num_patch = int((max_frames*spatial_size**2) / (2*16*16))
        self.nt = max_frames//2
        self.nh_nw = int((spatial_size/16)**2)
        self.positional_encoding = PositionalEncoding(sequence_length=num_patch, embed_dim=d_models)
        self.blocks = [EncoderBlock(d_models, num_heads, self.nt, self.nh_nw) for _ in range(num_l)]

    def call(self, inputs, training=True, mask=None):
        Z = self.patch_embedding(inputs)
        Z = layers.Add()([Z, self.positional_encoding(Z)])
        for block in self.blocks:
            Z = block(Z, mask=mask, training=training)
        return Z
