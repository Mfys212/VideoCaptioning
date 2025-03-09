import os
import tensorflow as tf
import numpy as np
from VideoCaptioning.data import *
from VideoCaptioning.evaluation import *
from VideoCaptioning.model import *
from VideoCaptioning.train import *

class CreateModel():
    def __init__(self, seed=False, multigpu=False):
        super().__init__()
        if seed:
            np.random.seed(212)
            tf.random.set_seed(212)
        self.model = self.train_data = self.test_data = None
        self.multigpu = multigpu
        self.D_MODELS = self.SEQ_LENGTH = self.VOCAB_SIZE = self.SPATIAL_SIZE = self.MAX_FRAMES = None
        self.FRAMES_STORAGE_PATH = self.NUM_CAPTIONS = self.BATCH_SIZE = None

    def LoadData(self, CAPTIONS_PATH, FRAMES_STORAGE_PATH, train_size, SEQ_LENGTH, 
                 VOCAB_SIZE, SPATIAL_SIZE, MAX_FRAMES, NUM_CAPTIONS, BATCH_SIZE, VIDEOS_PATH):
        captions_mapping, text_data = load_captions_data(CAPTIONS_PATH, SEQ_LENGTH, VIDEOS_PATH)
        train_data, valid_data = train_val_split(captions_mapping, train_size)
        vectorization = vectoriz_text(text_data, VOCAB_SIZE, SEQ_LENGTH)
        process_frames(FRAMES_STORAGE_PATH, captions_mapping, SPATIAL_SIZE, MAX_FRAMES)
        train_frame_dirs = [os.path.join(FRAMES_STORAGE_PATH, 
                                         os.path.basename(video).split('.')[0]) for video in train_data.keys()]
        valid_frame_dirs = [os.path.join(FRAMES_STORAGE_PATH, 
                                         os.path.basename(video).split('.')[0]) for video in valid_data.keys()]
        train_dataset = make_dataset_from_frames(train_frame_dirs, list(train_data.values()), 
                                                 vectorization, NUM_CAPTIONS, SPATIAL_SIZE, MAX_FRAMES, BATCH_SIZE)
        valid_dataset = make_dataset_from_frames(valid_frame_dirs, list(valid_data.values()), 
                                                 vectorization, NUM_CAPTIONS, SPATIAL_SIZE, MAX_FRAMES, BATCH_SIZE)
        self.train_data = train_dataset
        self.test_data = valid_dataset

    def DefineModel(self, ENCODER, DECODER, D_MODELS, NUM_HEADS, MAX_FRAMES, SPATIAL_SIZE, 
                    NUM_PATCH, VOCAB_SIZE, SEQ_LENGTH, NUM_L):
        encoder = ENCODER(d_models=D_MODELS, num_heads=NUM_HEADS, 
                                          num_l=NUM_L, max_frames=MAX_FRAMES)
        encoder.build(input_shape=(None, MAX_FRAMES, *SPATIAL_SIZE, 3))  
        decoder = DECODER(d_models=D_MODELS, num_heads=NUM_HEADS, 
                                vocab_size=VOCAB_SIZE, seq_length=SEQ_LENGTH, num_l=NUM_L)
        decoder.build([(None, None), (None, NUM_PATCH, D_MODELS), (None, None)])
        model = MainModel(encoder=encoder, decoder=decoder, num_captions_per_video=self.NUM_CAPTIONS)
        return encoder, decoder, model

    def SpatioTemporalAttention(self, D_MODELS, NUM_HEADS, MAX_FRAMES, SPATIAL_SIZE, 
                               VOCAB_SIZE, SEQ_LENGTH, NUM_L):
        NUM_PATCH = int((MAX_FRAMES*(SPATIAL_SIZE)**2) / (1*16**2))
        if self.multigpu == True:
            with strategy.scope():
                encoder, decoder, model = self.DefineModel(SpatioTemporalAttention.Encoder, module.Decoder, D_MODELS, 
                                                           NUM_HEADS, MAX_FRAMES, SPATIAL_SIZE, NUM_PATCH, VOCAB_SIZE, SEQ_LENGTH, NUM_L)
        else:
            encoder, decoder, model = self.DefineModel(SpatioTemporalAttention.Encoder, module.Decoder, D_MODELS, 
                                                       NUM_HEADS, MAX_FRAMES, SPATIAL_SIZE, NUM_PATCH, VOCAB_SIZE, SEQ_LENGTH, NUM_L)  
        self.D_MODELS = D_MODELS
        self.SEQ_LENGTH = SEQ_LENGTH
        self.VOCAB_SIZE = VOCAB_SIZE
        self.SPATIAL_SIZE = SPATIAL_SIZE
        self.MAX_FRAMES = MAX_FRAMES
        trainable_vars = model.trainable_variables
        total_params = 0
        for var in trainable_vars:
            var_params = tf.size(var).numpy()
            total_params += var_params
        print(f"Num of trainable parameters: {total_params}")
        self.model = model

    def FactorisedEncoder(self, D_MODELS, NUM_HEADS, MAX_FRAMES, SPATIAL_SIZE, 
                        VOCAB_SIZE, SEQ_LENGTH, NUM_L):
        NUM_PATCH = MAX_FRAMES
        if self.multigpu == True:
            with strategy.scope():
                encoder, decoder, model = self.DefineModel(FactorisedEncoder.Encoder, module.Decoder, D_MODELS, 
                                                           NUM_HEADS, MAX_FRAMES, SPATIAL_SIZE, NUM_PATCH, VOCAB_SIZE, SEQ_LENGTH, NUM_L)
        else:
            encoder, decoder, model = self.DefineModel(FactorisedEncoder.Encoder, module.Decoder, D_MODELS, 
                                                       NUM_HEADS, MAX_FRAMES, SPATIAL_SIZE, NUM_PATCH, VOCAB_SIZE, SEQ_LENGTH, NUM_L)  
        self.D_MODELS = D_MODELS
        self.SEQ_LENGTH = SEQ_LENGTH
        self.VOCAB_SIZE = VOCAB_SIZE
        self.SPATIAL_SIZE = SPATIAL_SIZE
        self.MAX_FRAMES = MAX_FRAMES
        trainable_vars = model.trainable_variables
        total_params = 0
        for var in trainable_vars:
            var_params = tf.size(var).numpy()
            total_params += var_params
        print(f"Num of trainable parameters: {total_params}")
        self.model = model

    def FactorisedSelfAttention(self, D_MODELS, NUM_HEADS, MAX_FRAMES, SPATIAL_SIZE, 
                               VOCAB_SIZE, SEQ_LENGTH, NUM_L):
        NUM_PATCH = int((MAX_FRAMES*(SPATIAL_SIZE)**2) / (1*16**2))
        if self.multigpu == True:
            with strategy.scope():
                encoder, decoder, model = self.DefineModel(FactorisedSelfAttention.Encoder, module.Decoder, D_MODELS, 
                                                           NUM_HEADS, MAX_FRAMES, SPATIAL_SIZE, NUM_PATCH, VOCAB_SIZE, SEQ_LENGTH, NUM_L)
        else:
            encoder, decoder, model = self.DefineModel(FactorisedSelfAttention.Encoder, module.Decoder, D_MODELS, 
                                                       NUM_HEADS, MAX_FRAMES, SPATIAL_SIZE, NUM_PATCH, VOCAB_SIZE, SEQ_LENGTH, NUM_L)  
        self.D_MODELS = D_MODELS
        self.SEQ_LENGTH = SEQ_LENGTH
        self.VOCAB_SIZE = VOCAB_SIZE
        self.SPATIAL_SIZE = SPATIAL_SIZE
        self.MAX_FRAMES = MAX_FRAMES
        trainable_vars = model.trainable_variables
        total_params = 0
        for var in trainable_vars:
            var_params = tf.size(var).numpy()
            total_params += var_params
        print(f"Num of trainable parameters: {total_params}")
        self.model = model

    def FactorisedDotProductAttention(self, D_MODELS, NUM_HEADS, MAX_FRAMES, SPATIAL_SIZE, 
                                    VOCAB_SIZE, SEQ_LENGTH, NUM_L):
        NUM_PATCH = int((MAX_FRAMES*(SPATIAL_SIZE)**2) / (1*16**2))
        if self.multigpu == True:
            with strategy.scope():
                encoder, decoder, model = self.DefineModel(FactorisedDotProductAttention.Encoder, module.Decoder, D_MODELS, 
                                                           NUM_HEADS, MAX_FRAMES, SPATIAL_SIZE, NUM_PATCH, VOCAB_SIZE, SEQ_LENGTH, NUM_L)
        else:
            encoder, decoder, model = self.DefineModel(FactorisedDotProductAttention.Encoder, module.Decoder, D_MODELS, 
                                                       NUM_HEADS, MAX_FRAMES, SPATIAL_SIZE, NUM_PATCH, VOCAB_SIZE, SEQ_LENGTH, NUM_L)  
        self.D_MODELS = D_MODELS
        self.SEQ_LENGTH = SEQ_LENGTH
        self.VOCAB_SIZE = VOCAB_SIZE
        self.SPATIAL_SIZE = SPATIAL_SIZE
        self.MAX_FRAMES = MAX_FRAMES
        trainable_vars = model.trainable_variables
        total_params = 0
        for var in trainable_vars:
            var_params = tf.size(var).numpy()
            total_params += var_params
        print(f"Num of trainable parameters: {total_params}")
        self.model = model

    def CrossAttention(self, D_MODELS, NUM_HEADS, MAX_FRAMES, SPATIAL_SIZE, 
                       VOCAB_SIZE, SEQ_LENGTH, NUM_L):
        NUM_PATCH = int((SPATIAL_SIZE/16)**2) + MAX_FRAMES
        if self.multigpu == True:
            strategy = tf.distribute.MirroredStrategy()
            with strategy.scope():
                encoder, decoder, model = self.DefineModel(CrossAttention.Encoder, module.Decoder, D_MODELS, 
                                                           NUM_HEADS, MAX_FRAMES, SPATIAL_SIZE, NUM_PATCH, VOCAB_SIZE, SEQ_LENGTH, NUM_L)
        else:
            encoder, decoder, model = self.DefineModel(CrossAttention.Encoder, module.Decoder, D_MODELS, 
                                                       NUM_HEADS, MAX_FRAMES, SPATIAL_SIZE, NUM_PATCH, VOCAB_SIZE, SEQ_LENGTH, NUM_L)  
        self.D_MODELS = D_MODELS
        self.SEQ_LENGTH = SEQ_LENGTH
        self.VOCAB_SIZE = VOCAB_SIZE
        self.SPATIAL_SIZE = SPATIAL_SIZE
        self.MAX_FRAMES = MAX_FRAMES
        trainable_vars = model.trainable_variables
        total_params = 0
        for var in trainable_vars:
            var_params = tf.size(var).numpy()
            total_params += var_params
        print(f"Num of trainable parameters: {total_params}")
        self.model = model

    def fit(self, CAPTIONS_PATH, VIDEOS_PATH, FRAMES_STORAGE_PATH, EPOCHS, BATCH_SIZE, NUM_CAPTIONS=40, train_size=0.8, VOCAB_SIZE=self.VOCAB_SIZE, SPATIAL_SIZE=self.SPATIAL_SIZE, MAX_FRAMES=self.MAX_FRAMES):
        if self.train_data is None or NUM_CAPTIONS != self.NUM_CAPTIONS or BATCH_SIZE != self.BATCH_SIZE or self.VOCAB_SIZE != VOCAB_SIZE or self.SPATIAL_SIZE != SPATIAL_SIZE or self.MAX_FRAMES != MAX_FRAMES or FRAMES_STORAGE_PATH != self.FRAMES_STORAGE_PATH:
            self.NUM_CAPTIONS = NUM_CAPTIONS
            self.BATCH_SIZE = BATCH_SIZE
            self.VOCAB_SIZE = VOCAB_SIZE
            self.SPATIAL_SIZE = SPATIAL_SIZE
            self.MAX_FRAMES = MAX_FRAMES
            self.FRAMES_STORAGE_PATH = FRAMES_STORAGE_PATH
            self.LoadData(CAPTIONS_PATH, self.FRAMES_STORAGE_PATH, train_size, self.SEQ_LENGTH, 
                         self.VOCAB_SIZE, self.SPATIAL_SIZE, self.MAX_FRAMES, self.NUM_CAPTIONS, self.BATCH_SIZE, VIDEOS_PATH)
        if self.multigpu == True:
            with strategy.scope():
                cross_entropy, early_stopping, optimizer = DefineCompile(self.D_MODELS)
                self.model.compile(optimizer=optimizer, loss=cross_entropy)
        else:
            cross_entropy, early_stopping, optimizer = DefineCompile(self.D_MODELS)
            self.model.compile(optimizer=optimizer, loss=cross_entropy)
            
        history = self.model.fit(self.train_data, 
                                 epochs=EPOCHS, 
                                 validation_data=self.test_data, 
                                 callbacks=[early_stopping])
        return history

    def eval(self):
        evaluation = EvalMetrics(sel.model, self.vectorization, self.SEQ_LENGTH, 
                                 self.test_data, self.FRAMES_STORAGE_PATH)
        acc, loss = evaluation.acc_loss()
        cider = evaluation.compute_cider()
        return acc, loss, cider
