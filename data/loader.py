import os
import re
import numpy as np
import cv2
from tqdm import tqdm
from glob import glob
from itertools import islice, cycle
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.layers import TextVectorization
from concurrent.futures import ThreadPoolExecutor, as_completed
import multiprocessing
try:
    from VideoCaptioning.main import set_seed
    set_seed()
except:
    pass
cpu_count = multiprocessing.cpu_count()

AUTOTUNE = tf.data.AUTOTUNE
exs = (".mp4", ".avi", ".mov", ".mkv", ".wmv", ".flv", ".webm", ".mpeg", ".mpg", ".3gp")

# Load captions data
def load_captions_data(filename, SEQ_LENGTH, VIDEOS_PATH):
    """Loads captions (text) data and maps them to corresponding videos."""
    with open(filename) as caption_file:
        caption_data = caption_file.readlines()
        caption_mapping = {}
        text_data = []
        videos_to_skip = set()

        for line in caption_data:
            line = line.rstrip("\n")
            parts = line.split(" ", 1)
            if len(parts) < 1:
                continue
            video_name, caption = parts
            caption = caption.strip()
            # Skip empty captions
            if not caption or caption == 0:
                continue
            tokens = caption.split()
            if len(tokens) < 1 or len(tokens) > SEQ_LENGTH:
                videos_to_skip.add(video_name)
                continue
            video_name = os.path.join(VIDEOS_PATH, video_name.strip() + '.avi')
            if video_name.endswith(exs) and video_name not in videos_to_skip:
                caption = "<start> " + caption + " <end>"
                text_data.append(caption)
                if video_name in caption_mapping:
                    caption_mapping[video_name].append(caption)
                else:
                    caption_mapping[video_name] = [caption]

        for video_name in videos_to_skip:
            if video_name in caption_mapping:
                del caption_mapping[video_name]

        return caption_mapping, text_data

# Split data into training and validation sets
def train_val_split(caption_data, train_size=0.8, shuffle=True):
    all_videos = list(caption_data.keys())
    if shuffle:
        np.random.shuffle(all_videos)
    train_size = int(len(caption_data) * train_size)
    training_data = {
        video_name: caption_data[video_name] for video_name in all_videos[:train_size]
    }
    validation_data = {
        video_name: caption_data[video_name] for video_name in all_videos[train_size:]
    }
    return training_data, validation_data

def vectoriz_text(text_data, VOCAB_SIZE, SEQ_LENGTH):
    def custom_standardization(input_string):
        return tf.strings.lower(input_string)
    
    vectorization = TextVectorization(
        max_tokens=VOCAB_SIZE,
        output_mode="int",
        output_sequence_length=SEQ_LENGTH,
        standardize=custom_standardization,
    )
    vectorization.adapt(text_data)
    return vectorization

def process_frames(FRAMES_STORAGE_PATH, captions_mapping, IMAGE_SIZE, MAX_FRAMES):
    def save_video_frames(video_path, output_dir, size=IMAGE_SIZE, max_frames=MAX_FRAMES):
        cap = cv2.VideoCapture(video_path)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        os.makedirs(output_dir, exist_ok=True)
        if total_frames <= max_frames:
            selected_frames = list(range(total_frames))
        else:
            selected_frames = np.linspace(0, total_frames - 1, max_frames, dtype=int)
        
        for idx, frame_idx in enumerate(selected_frames):
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
            ret, frame = cap.read()
            if not ret:
                break
            
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    #         frame = cv2.resize(frame, size)
            frame_filename = os.path.join(output_dir, f"frame_{idx:04d}.jpg")
            cv2.imwrite(frame_filename, frame)
    
        cap.release()
    
    if not os.path.exists(FRAMES_STORAGE_PATH):
        video_paths = list(captions_mapping.keys())
    
        with ThreadPoolExecutor(max_workers=cpu_count+2) as executor:
            list(tqdm(executor.map(
                lambda video_path: save_video_frames(video_path, os.path.join(FRAMES_STORAGE_PATH, os.path.basename(video_path).split('.')[0]),  ),
                video_paths
            ), total=len(video_paths), desc="Saving frames"))

def load_frames_from_directory(directory, size, max_frames):
    try:
        directory = directory.numpy().decode('utf-8')
    except:
        pass
    frame_files = sorted(glob(os.path.join(directory, "*.jpg")))
    frames = []
    for frame_file in frame_files[:max_frames]:
        frame = tf.io.read_file(frame_file)
        frame = tf.image.decode_jpeg(frame, channels=3)
        frame = tf.image.resize(frame, (size, size))
        frames.append(frame)

    if len(frames) < max_frames:
        padding = [tf.zeros((size, size, 3), dtype=tf.float32)] * (max_frames - len(frames))
        frames.extend(padding)

    video_tensor = tf.stack(frames, axis=0)
    return video_tensor

def tf_load_frames_from_directory(directory, IMAGE_SIZE, max_frames):
    video_tensor = tf.py_function(
        func=load_frames_from_directory,
        inp=[directory, IMAGE_SIZE, max_frames],
        Tout=tf.float32
    )
    
    video_tensor.set_shape((max_frames, IMAGE_SIZE, IMAGE_SIZE, 3))
    return video_tensor

def pad_captions(captions, max_captions):
    captions_unique = list(set(captions))

    if len(captions_unique) > max_captions:
        captions_padded = captions_unique[:max_captions]
    else:
        captions_padded = list(islice(cycle(captions_unique), max_captions))

    return captions_padded

def make_dataset_from_frames(frame_directories, captions, vectorization, num_captions, IMAGE_SIZE, max_frames, BATCH_SIZE, split="train"):
    # frame_dataset = tf.data.Dataset.from_tensor_slices(frame_directories, IMAGE_SIZE, max_frames).map(
    #     tf_load_frames_from_directory, num_parallel_calls=AUTOTUNE
    # )
    frame_dataset = tf.data.Dataset.from_tensor_slices(frame_directories).map(
        lambda directory: tf_load_frames_from_directory(directory, IMAGE_SIZE, max_frames),
        num_parallel_calls=AUTOTUNE
    )

    captions_padded = [pad_captions(caption, num_captions) for caption in captions]
    cap_dataset = tf.data.Dataset.from_tensor_slices(captions_padded).map(
        vectorization, num_parallel_calls=AUTOTUNE
    )

    dataset = tf.data.Dataset.zip((frame_dataset, cap_dataset))
    dataset = dataset.batch(BATCH_SIZE).shuffle(256).prefetch(AUTOTUNE)
    return dataset
