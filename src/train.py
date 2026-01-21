import os
import sys
import site

user_site = site.getusersitepackages()
if user_site and user_site not in sys.path:
    sys.path.insert(0, user_site)

import argparse
import numpy as np
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.models import load_model
import matplotlib.pyplot as plt
from model import DefectDetectionModel
from preprocessing import ImagePreprocessor
from utils import load_config, create_directories, save_model_info

def train_model(config, model_type='resnet50', train_dir=None, val_dir=None):
    model_builder = DefectDetectionModel(config)
    model = model_builder.get_model(model_type)
    
    preprocessor = ImagePreprocessor(config)
    
    if train_dir and val_dir:
        train_gen, val_gen = preprocessor.create_generators(
            train_dir, 
            val_dir, 
            config['model']['batch_size']
        )
    else:
        print("Error: train_dir and val_dir must be provided")
        return None
    
    model_dir = config['paths']['model_dir']
    create_directories([model_dir])
    
    checkpoint_path = os.path.join(model_dir, f"best_model_{model_type}.h5")
    
    callbacks = [
        ModelCheckpoint(
            checkpoint_path,
            monitor='val_loss',
            save_best_only=True,
            verbose=1
        ),
        EarlyStopping(
            monitor='val_loss',
            patience=10,
            restore_best_weights=True,
            verbose=1
        ),
        ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=5,
            min_lr=1e-7,
            verbose=1
        )
    ]
    
    history = model.fit(
        train_gen,
        epochs=config['model']['epochs'],
        validation_data=val_gen,
        callbacks=callbacks,
        verbose=1
    )
    
    model.save(os.path.join(model_dir, f"final_model_{model_type}.h5"))
    
    metrics = {
        'final_train_loss': history.history['loss'][-1],
        'final_train_accuracy': history.history['accuracy'][-1],
        'final_val_loss': history.history['val_loss'][-1],
        'final_val_accuracy': history.history['val_accuracy'][-1],
        'best_val_loss': min(history.history['val_loss']),
        'best_val_accuracy': max(history.history['val_accuracy'])
    }
    
    save_model_info(model, model_dir, metrics)
    
    plot_training_history(history, model_dir, model_type)
    
    return model, history

def plot_training_history(history, save_dir, model_type):
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    
    axes[0].plot(history.history['loss'], label='Train Loss')
    axes[0].plot(history.history['val_loss'], label='Val Loss')
    axes[0].set_title('Model Loss')
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Loss')
    axes[0].legend()
    
    axes[1].plot(history.history['accuracy'], label='Train Accuracy')
    axes[1].plot(history.history['val_accuracy'], label='Val Accuracy')
    axes[1].set_title('Model Accuracy')
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('Accuracy')
    axes[1].legend()
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, f'training_history_{model_type}.png'))
    plt.close()

def main():
    parser = argparse.ArgumentParser(description='Train defect detection model')
    parser.add_argument('--config', type=str, default='config/config.yaml', help='Path to config file')
    parser.add_argument('--model', type=str, default='resnet50', choices=['resnet50', 'mobilenet', 'efficientnet', 'custom'], help='Model type')
    parser.add_argument('--train_dir', type=str, required=True, help='Path to training data directory')
    parser.add_argument('--val_dir', type=str, required=True, help='Path to validation data directory')
    
    args = parser.parse_args()
    
    config = load_config(args.config)
    train_model(config, args.model, args.train_dir, args.val_dir)

if __name__ == '__main__':
    main()

