class EncoderBlock(tf.keras.layers.Layer):
    def __init__(self, d_models, num_heads, **kwargs):
        super().__init__(**kwargs)
        d = d_models // num_heads
        self.attention1 = layers.MultiHeadAttention(num_heads=num_heads, key_dim=d, dropout=0.1)
        self.attention2 = layers.MultiHeadAttention(num_heads=num_heads, key_dim=d, dropout=0.1)
        self.layernorm = [layers.Normalization() for _ in range(4)]
        self.dense = layers.Dense(d_models, activation="gelu")
        self.densel = layers.Dense(d_models)
        self.dropout = layers.Dropout(0.1)

    def call(self, Zt, Zs, mask=None, training=True):
        Zts = tf.concat([Zs, Zt], axis=1)
        attention1 = self.attention1(query=Zt, value=Zts, key=Zts, attention_mask=mask, training=training)
        Zt = self.layernorm[0](layers.Add()([Zt, attention1]))
        Zst = tf.concat([Zs, Zt], axis=1)
        attention2 = self.attention2(query=Zs, value=Zst, key=Zst, attention_mask=mask, training=training)
        Zs = self.layernorm[1](layers.Add()([Zs, attention2]))

        ffn1 = self.densel(self.dropout(self.dense(Zt), training=training))
        Zt = self.layernorm[2](layers.Add()([ffn1, Zt]))
        ffn2 = self.densel(self.dropout(self.dense(Zs), training=training))
        Zs = self.layernorm[3](layers.Add()([ffn2, Zs]))
        return [Zt, Zs]
