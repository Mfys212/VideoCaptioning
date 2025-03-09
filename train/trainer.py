import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

class MainModel(keras.Model):
    def __init__(self, encoder, decoder, num_captions_per_video):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.loss_tracker = keras.metrics.Mean(name="loss")
        self.acc_tracker = keras.metrics.Mean(name="accuracy")
        self.num_captions_per_video = num_captions_per_video

    def calculate_loss(self, y_true, y_pred, mask):
        loss = self.loss(y_true, y_pred)
        mask = tf.cast(mask, dtype=loss.dtype)
        loss *= mask
        return tf.reduce_sum(loss) / tf.reduce_sum(mask)

    def calculate_accuracy(self, y_true, y_pred, mask):
        accuracy = tf.equal(y_true, tf.argmax(y_pred, axis=2))
        accuracy = tf.math.logical_and(mask, accuracy)
        accuracy = tf.cast(accuracy, dtype=tf.float32)
        mask = tf.cast(mask, dtype=tf.float32)
        return tf.reduce_sum(accuracy) / tf.reduce_sum(mask)

    def _compute_caption_loss_and_acc(self, encoder_out, batch_seq, training=True):
        batch_seq_inp = batch_seq[:, :-1]
        batch_seq_true = batch_seq[:, 1:]
        mask = tf.math.not_equal(batch_seq_true, 0)
        batch_seq_pred = self.decoder(
            [batch_seq_inp, encoder_out, mask], training=training
        )
        loss = self.calculate_loss(batch_seq_true, batch_seq_pred, mask)
        acc = self.calculate_accuracy(batch_seq_true, batch_seq_pred, mask)
        return loss, acc

    def _compute_loss_and_acc_for_captions(self, encoder_out, batch_seq, training):
        def compute_loss_acc(i):
            return self._compute_caption_loss_and_acc(
                encoder_out, batch_seq[:, i, :], training=training
            )
        loss_acc_pairs = tf.map_fn(compute_loss_acc, tf.range(self.num_captions_per_video), 
                                   dtype=(tf.float32, tf.float32))
        batch_loss = tf.reduce_sum(loss_acc_pairs[0])
        batch_acc = tf.reduce_mean(loss_acc_pairs[1])
        return batch_loss, batch_acc

    def train_step(self, batch_data):
        batch_video, batch_seq = batch_data
        batch_loss = batch_acc = 0
        with tf.GradientTape() as tape:
            encoder_out = self.encoder(batch_video, training=True)
            batch_loss, batch_acc = self._compute_loss_and_acc_for_captions(encoder_out, 
                                                                            batch_seq, training=True)      
        train_vars = (
            self.encoder.trainable_variables + 
            self.decoder.trainable_variables
        )
        grads = tape.gradient(batch_loss, train_vars)
        self.optimizer.apply_gradients(zip(grads, train_vars))
        self.loss_tracker.update_state(batch_loss)
        self.acc_tracker.update_state(batch_acc)
        return {"seq_loss": self.loss_tracker.result(), "seq_acc": self.acc_tracker.result()}

    def test_step(self, batch_data):
        batch_video, batch_seq = batch_data
        batch_loss = batch_acc = 0
        encoder_out = self.encoder(batch_video, training=False)
        batch_loss, batch_acc = self._compute_loss_and_acc_for_captions(encoder_out, batch_seq, training=False)
        self.loss_tracker.update_state(batch_loss)
        self.acc_tracker.update_state(batch_acc)
        return {"seq_loss": self.loss_tracker.result(), "seq_acc": self.acc_tracker.result()}

    @property
    def metrics(self):
        return [self.loss_tracker, self.acc_tracker]

def DefineCompile(D_MODELS):
    cross_entropy = keras.losses.SparseCategoricalCrossentropy(
        from_logits=True, reduction="none"
    )
    early_stopping = keras.callbacks.EarlyStopping(patience=15, restore_best_weights=True)
    class CustomSchedule(tf.keras.optimizers.schedules.LearningRateSchedule):
        def __init__(self, d_model, warmup_steps=4000):
            super(CustomSchedule, self).__init__()
            self.d_model = tf.cast(d_model, tf.float32)
            self.warmup_steps = warmup_steps
    
        def __call__(self, step):
            step = tf.cast(step, tf.float32)
            arg1 = tf.math.rsqrt(step)
            arg2 = step * (self.warmup_steps ** -1.5)
            return tf.math.rsqrt(self.d_model) * tf.math.minimum(arg1, arg2)
    
    lr_schedule = CustomSchedule(D_MODELS)
    optimizer = tf.keras.optimizers.Adam(
        learning_rate=lr_schedule,
        beta_1=0.9,                 
        beta_2=0.98,                
        epsilon=1e-9                
    )
    return cross_entropy, early_stopping, optimizer
