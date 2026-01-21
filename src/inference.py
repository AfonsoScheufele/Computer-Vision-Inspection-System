import os
import sys
import site

user_site = site.getusersitepackages()
if user_site and user_site not in sys.path:
    sys.path.insert(0, user_site)

import argparse
import numpy as np
import cv2
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model
from preprocessing import ImagePreprocessor
from utils import load_config

class DefectDetector:
    def __init__(self, model_path, config_path='config/config.yaml'):
        self.config = load_config(config_path)
        self.model = load_model(model_path)
        self.preprocessor = ImagePreprocessor(self.config)
        
    def predict(self, image_path):
        image = self.preprocessor.preprocess_image(image_path)
        image_batch = np.expand_dims(image, axis=0)
        
        prediction = self.model.predict(image_batch, verbose=0)
        
        if self.config['model']['num_classes'] == 1:
            defect_probability = prediction[0][0]
            is_defect = defect_probability > 0.5
            return {
                'is_defect': bool(is_defect),
                'defect_probability': float(defect_probability),
                'confidence': float(defect_probability if is_defect else 1 - defect_probability)
            }
        else:
            class_idx = np.argmax(prediction[0])
            confidence = float(prediction[0][class_idx])
            return {
                'class': int(class_idx),
                'confidence': float(confidence),
                'probabilities': prediction[0].tolist()
            }
    
    def predict_batch(self, image_paths):
        results = []
        for image_path in image_paths:
            try:
                result = self.predict(image_path)
                result['image_path'] = image_path
                results.append(result)
            except Exception as e:
                print(f"Error processing {image_path}: {e}")
                continue
        return results
    
    def visualize_prediction(self, image_path, save_path=None):
        original_image = cv2.imread(image_path)
        original_image = cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB)
        
        prediction = self.predict(image_path)
        
        fig, axes = plt.subplots(1, 2, figsize=(12, 6))
        
        axes[0].imshow(original_image)
        axes[0].set_title('Original Image')
        axes[0].axis('off')
        
        if self.config['model']['num_classes'] == 1:
            status = "DEFECT" if prediction['is_defect'] else "NORMAL"
            color = 'red' if prediction['is_defect'] else 'green'
            axes[1].text(0.5, 0.5, f"{status}\nConfidence: {prediction['confidence']:.2%}", 
                        ha='center', va='center', fontsize=20, color=color, 
                        transform=axes[1].transAxes, weight='bold')
        else:
            class_names = ['Normal', 'Defect']
            class_name = class_names[prediction['class']]
            axes[1].text(0.5, 0.5, f"Class: {class_name}\nConfidence: {prediction['confidence']:.2%}", 
                        ha='center', va='center', fontsize=20, 
                        transform=axes[1].transAxes, weight='bold')
        
        axes[1].axis('off')
        axes[1].set_title('Prediction Result')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path)
        else:
            plt.show()
        
        plt.close()
        
        return prediction

def main():
    parser = argparse.ArgumentParser(description='Run inference on images')
    parser.add_argument('--model', type=str, required=True, help='Path to trained model')
    parser.add_argument('--image', type=str, help='Path to single image')
    parser.add_argument('--dir', type=str, help='Path to directory of images')
    parser.add_argument('--output', type=str, help='Output directory for results')
    parser.add_argument('--config', type=str, default='config/config.yaml', help='Path to config file')
    parser.add_argument('--visualize', action='store_true', help='Generate visualization')
    
    args = parser.parse_args()
    
    detector = DefectDetector(args.model, args.config)
    
    if args.image:
        result = detector.predict(args.image)
        print(f"Image: {args.image}")
        print(f"Result: {result}")
        
        if args.visualize:
            output_path = args.output if args.output else None
            detector.visualize_prediction(args.image, output_path)
    
    elif args.dir:
        image_files = [os.path.join(args.dir, f) for f in os.listdir(args.dir) 
                      if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
        
        results = detector.predict_batch(image_files)
        
        if args.output:
            os.makedirs(args.output, exist_ok=True)
            for result in results:
                if args.visualize:
                    image_name = os.path.basename(result['image_path'])
                    save_path = os.path.join(args.output, f"result_{image_name}")
                    detector.visualize_prediction(result['image_path'], save_path)
                print(f"{result['image_path']}: {result}")
        else:
            for result in results:
                print(f"{result['image_path']}: {result}")
    else:
        print("Error: Either --image or --dir must be provided")

if __name__ == '__main__':
    main()

