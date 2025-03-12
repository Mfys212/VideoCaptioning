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
    def __init__(self, d_models, num_heads, max_frames, spatial_size, **kwargs):
        super().__init__(**kwargs)
        d = d_models // num_heads
        self.attention1 = layers.MultiHeadAttention(num_heads=num_heads, key_dim=d, dropout=0.1)
        self.attention2 = layers.MultiHeadAttention(num_heads=num_heads, key_dim=d, dropout=0.1)
        self.layernorm = [layers.LayerNormalization() for _ in range(3)]
        self.dense = layers.Dense(d_models, activation="gelu")
        self.densel = layers.Dense(d_models)
        self.dropout = layers.Dropout(0.1)
        self.nt = max_frames//2
        self.nh_nw = int((spatial_size/16)**2)

    def call(self, Z, mask=None, training=True):
        batch_size = tf.shape(Z)[0]
        Z = tf.reshape(Z, (batch_size, self.nt, self.nh_nw, self.d_model))
        Zs = tf.reshape(Z, (batch_size*self.nt, self.nh_nw, self.d_model))
        Znorm = self.layernorm[0](Zs)
        attention = self.attention1(query=Znorm, value=Znorm, key=Znorm, attention_mask=mask, training=training)
        Zs = layers.Add()([Zs, attention])
        Z = tf.reshape(Zs, (batch_size, self.nt, self.nh_nw, self.d_model))
        Z = tf.transpose(Z, perm=[0, 2, 1, 3])
        Zt = tf.reshape(Z, (batch_size*self.nh_nw, self.nt, self.d_model))
        Znorm = self.layernorm[1](Zt)
        attention = self.attention2(query=Znorm, value=Znorm, key=Znorm, attention_mask=mask, training=training)
        Zt = layers.Add()([Zt, attention])
        ffn = self.densel(self.dropout(self.dense(self.layernorm[2](Zt)), training=training))
        Z = layers.Add()([Zt, ffn])
        return tf.reshape(Z, (batch_size, self.nt * self.nh_nw, self.d_model))

class Encoder(tf.keras.models.Model):
    def __init__(self, d_models, num_heads, num_l, max_frames, spatial_size, **kwargs):
        super().__init__(**kwargs)
        self.patch_embedding = PatchEmbedding(d_models, 2, 16, 16)
        num_patch = int((max_frames*spatial_size**2) / (2*16*16))
        self.Spositional_encoding = PositionalEncoding(sequence_length=num_patch, embed_dim=d_models)
        self.blocks = [EncoderBlock(d_models, num_heads, max_frames, spatial_size) for _ in range(num_l)]

    def call(self, inputs, training=True, mask=None):
        Z = self.patch_embedding(inputs)
        Z = layers.Add()([Z, self.positional_encoding(Z)])
        for block in self.blocks:
            Z = block(Z, mask=mask, training=training)
        return Z
