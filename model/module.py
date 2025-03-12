import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
try:
    from VideoCaptioning import set_seed
    set_seed()
except:
    pass

class PatchEmbedding(layers.Layer):
    def __init__(self, embed_dim, patch_tem, patch_height, patch_width, **kwargs):
        super().__init__(**kwargs)
        self.projection = layers.Conv3D(
            filters=embed_dim,
            kernel_size=(patch_tem, patch_height, patch_width), 
            strides=(patch_tem, patch_height, patch_width), 
            padding="VALID",
        )
        self.flatten = layers.Reshape(target_shape=(-1, embed_dim))

    def call(self, videos):
        projected_patches = self.projection(videos)
        flattened_patches = self.flatten(projected_patches)
        return flattened_patches

class PatchEmbedding2(layers.Layer):
    def __init__(self, embed_dim, patch_tem, patch_height, patch_width, **kwargs):
        super().__init__(**kwargs)
        self.projection = layers.Conv3D(
            filters=embed_dim,
            kernel_size=(patch_tem, patch_height, patch_width), 
            strides=(patch_tem, patch_height, patch_width), 
            padding="VALID",
        )

    def call(self, videos):
        projected_patches = self.projection(videos)
        return projected_patches

class PositionalEncoding(tf.keras.layers.Layer):
    def __init__(self, sequence_length, embed_dim, **kwargs):
        super().__init__(**kwargs)
        self.positional_encoding = self.add_weight(
            name="positional_encoding",
            shape=(sequence_length, embed_dim), 
            initializer=tf.keras.initializers.RandomNormal(),
            trainable=True
        )

    def call(self, inputs):
        batch_size = tf.shape(inputs)[0] 
        length = tf.shape(inputs)[1]  
        pos_encoding = tf.expand_dims(self.positional_encoding[:length, :], axis=0)  
        return tf.tile(pos_encoding, [batch_size, 1, 1])  

class DotProductAttention(layers.Layer):
    def __init__(self, **kwargs):
        super(DotProductAttention, self).__init__(**kwargs)

    def call(self, queries, keys, values, mask=None):
        d_k = tf.sqrt(tf.cast(tf.shape(keys)[-1], keys.dtype)) 
        scores = tf.einsum('...qd,...kd->...qk', queries, keys) / d_k 
        if mask is not None:
            scores += -1e9 * mask
        attention_weights = tf.nn.softmax(scores, axis=-1)
        return tf.einsum('...qk,...kv->...qv', attention_weights, values) 

class MMultiHeadAttention(layers.Layer):
    def __init__(self, num_heads, key_dim, d_models, nt, nh_nw, dropout=0.1, **kwargs):
        super(MMultiHeadAttention, self).__init__(**kwargs)
        assert num_heads % 2 == 0, "num_heads must be even for division into two parts"
        self.num_heads = num_heads
        self.half_heads = num_heads // 2
        self.key_dim = key_dim
        self.d_model = d_models
        self.attention = DotProductAttention()
        self.W_q = layers.Dense(d_models)   
        self.W_kt = layers.Dense(d_models//2)
        self.W_ks = layers.Dense(d_models//2)
        self.W_vt = layers.Dense(d_models//2)
        self.W_vs = layers.Dense(d_models//2)
        self.W_o = layers.Dense(d_models)  
        self.dropout = layers.Dropout(dropout)
        self.nt = nt
        self.nh_nw = nh_nw

    def reshape_tensor(self, x, heads):
        batch_size = tf.shape(x)[0]
        seq_len = tf.shape(x)[1]
        x = tf.reshape(x, (batch_size, seq_len, heads, self.key_dim))
        return tf.transpose(x, perm=[0, 2, 1, 3])

    def call(self, query, keys, values, mask=None, training=True):
        batch_size = tf.shape(query)[0]
        st = tf.reshape(query, (batch_size, self.nt, self.nh_nw, self.d_model))
        t = tf.reduce_mean(st, axis=2)
        s = tf.reduce_mean(st, axis=1)
        q = self.W_q(query)
        k1 = self.W_ks(s) 
        v1 = self.W_vs(s) 
        k2 = self.W_kt(t) 
        v2 = self.W_vt(t) 
        
        q_heads = self.reshape_tensor(q, self.num_heads)  
        k_heads_1 = self.reshape_tensor(k1, self.half_heads)  
        k_heads_2 = self.reshape_tensor(k2, self.half_heads)  
        v_heads_1 = self.reshape_tensor(v1, self.half_heads)  
        v_heads_2 = self.reshape_tensor(v2, self.half_heads) 
        # q_heads_1, q_heads_2 = tf.split(q_heads, num_or_size_splits=2, axis=1)
        # attn_out_1 = self.attention(q_heads_1, k_heads_1, v_heads_1, mask)
        # attn_out_2 = self.attention(q_heads_2, k_heads_2, v_heads_2, mask)
        # attn_out_1 = self.attention(q_heads[:, :self.half_heads], k_heads_1, v_heads_1, mask)
        # attn_out_2 = self.attention(q_heads[:, self.half_heads:], k_heads_2, v_heads_2, mask)
        # attn_output = tf.concat([attn_out_1, attn_out_2], axis=1)
        k_heads = tf.concat([k_heads_1, k_heads_2], axis=1) 
        v_heads = tf.concat([v_heads_1, v_heads_2], axis=1) 
        attn_output = self.attention(q_heads, k_heads, v_heads, mask)
        attn_output = tf.transpose(attn_output, perm=[0, 2, 1, 3]) 
        attn_output = tf.reshape(attn_output, (batch_size, -1, self.d_model))
        return self.dropout(self.W_o(attn_output), training=training)

ubah semuanya (yang bisa) menggunakan einsum 

class TransformerBlock(tf.keras.layers.Layer):
    def __init__(self, d_models, num_heads, **kwargs):
        super().__init__(**kwargs)
        d = d_models // num_heads
        self.attention = layers.MultiHeadAttention(num_heads=num_heads, key_dim=d, dropout=0.1)
        self.layernorm_1 = layers.LayerNormalization()
        self.layernorm_2 = layers.LayerNormalization()
        self.dense = layers.Dense(d_models, activation="gelu")
        self.densel = layers.Dense(d_models)

    def call(self, query, key, value, mask=None, training=True):
        attention_output = self.attention(query=query, key=key, value=value, attention_mask=mask, training=training)
        out_1 = self.layernorm_1(layers.Add()([query, attention_output]))
        inputs = self.densel(self.dense(out_1))
        out = self.layernorm_2(layers.Add()([inputs, out_1]))
        return out

class Decoder(tf.keras.models.Model):
    def __init__(self, d_models, num_heads, vocab_size, seq_length, num_l, **kwargs):
        super().__init__(**kwargs)
        self.token_embeddings = layers.Embedding(input_dim=vocab_size, output_dim=d_models)
        self.embed_scale = tf.math.sqrt(tf.cast(d_models, tf.float32))
        self.positional_encoding = PositionalEncoding(sequence_length=seq_length, embed_dim=d_models)
        self.mask_att = layers.MultiHeadAttention(
            num_heads=num_heads, key_dim=d_models // num_heads, dropout=0.1
        )
        self.layernorm = layers.LayerNormalization()
        self.attention = [TransformerBlock(d_models, num_heads) for _ in range(num_l)]
        self.linear = layers.Dense(d_models)
        self.out = layers.Dense(vocab_size)
    
    def call(self, inp, training=True):
        inputs, en_out, mask = inp
        inputs = self.token_embeddings(inputs)
        inputs = layers.Add()([inputs * self.embed_scale, self.positional_encoding(inputs)])
        causal_mask = self.get_causal_attention_mask(inputs)
        if mask is not None:
            padding_mask = tf.cast(mask[:, :, tf.newaxis], dtype=tf.int32)
            combined_mask = tf.cast(mask[:, tf.newaxis, :], dtype=tf.int32)
            combined_mask = tf.minimum(combined_mask, causal_mask)
        mask_out = self.mask_att(
            query=inputs, key=inputs, value=inputs, attention_mask=combined_mask, training=training
        )
        inputs = self.layernorm(layers.Add()([inputs, mask_out]))
        for layer in self.attention:
            inputs = layer(query=inputs, key=en_out, value=en_out, mask=padding_mask, training=training)
        return self.out(self.linear(inputs))

    def get_causal_attention_mask(self, inputs):
        input_shape = tf.shape(inputs)
        batch_size, sequence_length = input_shape[0], input_shape[1]
        i = tf.range(sequence_length)[:, tf.newaxis]
        j = tf.range(sequence_length)
        mask = tf.cast(i >= j, dtype="int32")
        mask = tf.reshape(mask, (1, input_shape[1], input_shape[1]))
        mult = tf.concat(
            [tf.expand_dims(batch_size, -1), tf.constant([1, 1], dtype=tf.int32)],
            axis=0,
        )
        return tf.tile(mask, mult)
