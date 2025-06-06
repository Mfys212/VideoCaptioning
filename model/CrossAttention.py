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
    def __init__(self, d_models, num_heads, TEMPORAL_FIRST=True, **kwargs):
        super().__init__(**kwargs)
        d = d_models // num_heads
        self.attention1 = layers.MultiHeadAttention(num_heads=num_heads, key_dim=d, dropout=0.1)
        self.attention2 = layers.MultiHeadAttention(num_heads=num_heads, key_dim=d, dropout=0.1)
        self.layernorm = [layers.LayerNormalization() for _ in range(4)]
        self.dense = layers.Dense(d_models, activation="gelu")
        self.densel = layers.Dense(d_models)
        self.dropout = layers.Dropout(0.1)
        self.tem_first = TEMPORAL_FIRST

    def call(self, inputs, mask=None, training=True):
        Zt, Zs = inputs
        if self.tem_first:
            Zts = tf.concat([Zs, Zt], axis=1)
            attention1 = self.attention1(query=Zt, value=Zts, key=Zts, attention_mask=mask, training=training)
            Zt = self.layernorm[0](layers.Add()([Zt, attention1]))
            Zst = tf.concat([Zs, Zt], axis=1)
            attention2 = self.attention2(query=Zs, value=Zst, key=Zst, attention_mask=mask, training=training)
            Zs = self.layernorm[1](layers.Add()([Zs, attention2]))
        else:
            Zts = tf.concat([Zs, Zt], axis=1)
            attention1 = self.attention1(query=Zs, value=Zts, key=Zts, attention_mask=mask, training=training)
            Zs = self.layernorm[0](layers.Add()([Zs, attention1]))
            Zst = tf.concat([Zs, Zt], axis=1)
            attention2 = self.attention2(query=Zt, value=Zst, key=Zst, attention_mask=mask, training=training)
            Zt = self.layernorm[1](layers.Add()([Zt, attention2]))

        ffn1 = self.densel(self.dropout(self.dense(Zt), training=training))
        Zt = self.layernorm[2](layers.Add()([ffn1, Zt]))
        ffn2 = self.densel(self.dropout(self.dense(Zs), training=training))
        Zs = self.layernorm[3](layers.Add()([ffn2, Zs]))
        return [Zt, Zs]

class Encoder(tf.keras.models.Model):
    def __init__(self, d_models, num_heads, num_l, max_frames, spatial_size, R_SPATIAL=4, R_TEMPORAL=4, F_SPATIAL=4, F_TEMPORAL=4, TEMPORAL_FIRST=True, **kwargs):
        super().__init__(**kwargs)
        self.tconv1 = layers.Conv3D(F_SPATIAL, (1, int(R_SPATIAL//2)+1, int(R_SPATIAL//2)+1), strides=(1, int(R_SPATIAL//2), int(R_SPATIAL//2)), padding='same')
        self.tconv2 = layers.Conv3D(3, (2, int(R_SPATIAL//2)+1, int(R_SPATIAL//2)+1), strides=(2, 2, 2), padding='same')
        self.flatten = layers.TimeDistributed(layers.Flatten())
        self.linear = layers.Dense(d_models)
        self.sconv1 = layers.Conv3D(F_TEMPORAL, (int(R_TEMPORAL//2)+1, 1, 1), strides=(int(R_TEMPORAL//2), 1, 1), padding='same')
        self.sconv2 = layers.Conv3D(3, (int(R_TEMPORAL//2)+1, 1, 1), strides=(2, 1, 1), padding='same')
        self.patch_embedding = PatchEmbedding(d_models, int(max_frames//R_TEMPORAL), 16, 16)
        num_patch = int((max_frames*spatial_size**2) / (max_frames*16*16))
        self.t_positional_encoding = PositionalEncoding(sequence_length=max_frames//2, embed_dim=d_models)
        self.s_positional_encoding = PositionalEncoding(sequence_length=num_patch, embed_dim=d_models)
        self.blocks = [EncoderBlock(d_models, num_heads, TEMPORAL_FIRST) for _ in range(num_l)]
        self.out = layers.Dense(d_models, activation="gelu")

    def call(self, inputs, training=True, mask=None):
        tem = self.linear(self.flatten(self.tconv2(self.tconv1(inputs))))
        Zt = layers.Add()([tem, self.t_positional_encoding(tem)])
        spa = self.patch_embedding(self.sconv2(self.sconv1(inputs)))
        Zs = layers.Add()([spa, self.s_positional_encoding(spa)])
        for block in self.blocks:
            Zt, Zs = block([Zt, Zs], mask=mask, training=training)
        out = tf.concat([Zs, Zt], axis=1)
        out = self.out(out)
        return out
