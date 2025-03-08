class EncoderBlock(tf.keras.layers.Layer):
    def __init__(self, d_models, num_heads, **kwargs):
        super().__init__(**kwargs)
        d = d_models // num_heads
        self.attention1 = layers.MultiHeadAttention(num_heads=num_heads, key_dim=d, dropout=0.1)
        self.attention2 = layers.MultiHeadAttention(num_heads=num_heads, key_dim=d, dropout=0.1)
        self.layernorm = [layers.Normalization() for _ in range(3)]
        self.dense = layers.Dense(d_models, activation="gelu")
        self.densel = layers.Dense(d_models)
        self.dropout = layers.Dropout(0.1)

    def call(self, Z, mask=None, training=True):
        for i in range(len(Z)):
            z = self.layernorm[0]([Z[i])
            attention = self.attention1(query=z, value=z, key=z, attention_mask=mask, training=training)
            Z[i] = layers.Add()([Z[i], attention])
        Z = tf.stack(Z, axis=1)
        num_patch = int(tf.shape(Z)[2])
        Z = tf.split(Z, num_or_size_splits=num_patch, axis=2)
        for i in range(len(Z)):
            z = self.layernorm[1]([Z[i])
            attention = self.attention2(query=z, value=z, key=z, attention_mask=mask, training=training)
            z = layers.Add()([Z[i], attention])
            ffn = self.densel(self.dropout(self.dense(self.layernorm[2](z)), training=training))
            Z[i] = layers.Add()([z, ffn])
        Z = tf.concat(Z, axis=1)
        return Z

class Encoder(tf.keras.Model):
    def __init__(self, d_models, num_heads, num_l, max_frames, spatial_size, **kwargs):
        super().__init__(**kwargs)
        self.patch_embedding = PatchEmbedding(d_models, 1, 16, 16)
        self.spa_num_patch = int((spatial_size/16)**2)
        self.Spositional_encoding = PositionalEncoding(sequence_length=self.spa_num_patch, embed_dim=d_models)
        self.Tpositional_encoding = PositionalEncoding(sequence_length=max_frames, embed_dim=d_models)
        self.blocks = [EncoderBlock(d_models, num_heads) for _ in range(num_l)]
        self.max_frames = max_frames
        self.d_models = d_models

    def call(self, inputs, training=True, mask=None):
        Z = tf.split(inputs, num_or_size_splits=self.max_frames, axis=1)
        Z = [self.patch_embedding(z) for z in Z]
        Z = [layers.Add()([z, self.Spositional_encoding(z)]) for z in Z]
        for block in self.encoder_blocks:
            Z = block(Z, training, mask)
            Z = tf.split(Z, num_or_size_splits=self.max_frames, axis=1)
        Z = tf.reshape(tf.concat(Z, axis=2), (int(tf.shape(inputs)[0]), self.max_frames * self.spa_num_patch, self.d_models))
        return Z
