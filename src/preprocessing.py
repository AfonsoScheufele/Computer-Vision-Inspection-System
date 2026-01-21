import os
import sys
import site

user_site = site.getusersitepackages()
if user_site and user_site not in sys.path:
    sys.path.insert(0, user_site)

import cv2
import numpy as np
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from pathlib import Path

class ImagePreprocessor:
    def __init__(self, config):
        self.config = config
        self.resize_size = tuple(config['preprocessing']['resize'])
        self.normalize = config['preprocessing']['normalize']
        self.grayscale = config['preprocessing']['grayscale']
        
    def load_image(self, image_path):
        if not os.path.exists(image_path):
            raise FileNotFoundError(f"Image not found: {image_path}")
        
        image = cv2.imread(image_path)
        if image is None:
            raise ValueError(f"Could not read image: {image_path}")
        
        if self.grayscale:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
        else:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        return image
    
    def resize_image(self, image):
        return cv2.resize(image, self.resize_size)
    
    def normalize_image(self, image):
        if self.normalize:
            return image.astype(np.float32) / 255.0
        return image.astype(np.float32)
    
    def preprocess_image(self, image_path):
        image = self.load_image(image_path)
        image = self.resize_image(image)
        image = self.normalize_image(image)
        return image
    
    def preprocess_batch(self, image_paths):
        images = []
        for path in image_paths:
            try:
                image = self.preprocess_image(path)
                images.append(image)
            except Exception as e:
                print(f"Error processing {path}: {e}")
                continue
        return np.array(images)
    
    def get_augmentation_generator(self):
        aug_config = self.config['augmentation']
        return ImageDataGenerator(
            rotation_range=aug_config['rotation_range'],
            width_shift_range=aug_config['width_shift_range'],
            height_shift_range=aug_config['height_shift_range'],
            horizontal_flip=aug_config['horizontal_flip'],
            zoom_range=aug_config['zoom_range'],
            brightness_range=aug_config['brightness_range'],
            fill_mode='nearest'
        )
    
    def create_generators(self, train_dir, val_dir, batch_size):
        train_datagen = self.get_augmentation_generator()
        val_datagen = ImageDataGenerator()
        
        train_generator = train_datagen.flow_from_directory(
            train_dir,
            target_size=self.resize_size[:2],
            batch_size=batch_size,
            class_mode='binary',
            shuffle=True
        )
        
        val_generator = val_datagen.flow_from_directory(
            val_dir,
            target_size=self.resize_size[:2],
            batch_size=batch_size,
            class_mode='binary',
            shuffle=False
        )
        
        return train_generator, val_generator

