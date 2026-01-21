import numpy as np
from PIL import Image, ImageDraw
import os
from pathlib import Path

def create_normal_image(size=(224, 224)):
    base_color = np.random.randint(200, 255, 3)
    image = np.full((*size, 3), base_color, dtype=np.uint8)
    
    noise = np.random.normal(0, 10, (*size, 3))
    image = np.clip(image.astype(np.float32) + noise, 0, 255).astype(np.uint8)
    
    img = Image.fromarray(image)
    draw = ImageDraw.Draw(img)
    
    draw.rectangle([50, 50, 174, 174], fill=(180, 180, 180), outline=(150, 150, 150), width=2)
    
    return np.array(img)

def create_defect_image(size=(224, 224)):
    base_color = np.random.randint(200, 255, 3)
    image = np.full((*size, 3), base_color, dtype=np.uint8)
    
    noise = np.random.normal(0, 15, (*size, 3))
    image = np.clip(image.astype(np.float32) + noise, 0, 255).astype(np.uint8)
    
    img = Image.fromarray(image)
    draw = ImageDraw.Draw(img)
    
    draw.rectangle([50, 50, 174, 174], fill=(180, 180, 180), outline=(150, 150, 150), width=2)
    
    defect_type = np.random.choice(['crack', 'scratch', 'hole', 'stain'])
    
    if defect_type == 'crack':
        points = [(100, 60), (120, 80), (140, 100), (160, 120)]
        draw.line(points, fill=(50, 50, 50), width=3)
    elif defect_type == 'scratch':
        draw.line([(70, 100), (150, 100)], fill=(80, 80, 80), width=4)
    elif defect_type == 'hole':
        draw.ellipse([97, 97, 127, 127], fill=(100, 100, 100))
    elif defect_type == 'stain':
        draw.ellipse([92, 92, 132, 132], fill=(120, 120, 120), outline=(100, 100, 100), width=2)
    
    return np.array(img)

def generate_dataset(output_dir, num_images=100, image_type='normal'):
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    for i in range(num_images):
        if image_type == 'normal':
            image = create_normal_image()
        else:
            image = create_defect_image()
        
        img = Image.fromarray(image)
        filename = f"{image_type}_{i:04d}.jpg"
        filepath = os.path.join(output_dir, filename)
        img.save(filepath, 'JPEG')
    
    print(f"Generated {num_images} {image_type} images in {output_dir}")

def main():
    print("Generating synthetic dataset...")
    print("=" * 50)
    
    train_normal_dir = "data/train/normal"
    train_defect_dir = "data/train/defect"
    val_normal_dir = "data/val/normal"
    val_defect_dir = "data/val/defect"
    
    print("\nGenerating training data...")
    generate_dataset(train_normal_dir, num_images=80, image_type='normal')
    generate_dataset(train_defect_dir, num_images=80, image_type='defect')
    
    print("\nGenerating validation data...")
    generate_dataset(val_normal_dir, num_images=20, image_type='normal')
    generate_dataset(val_defect_dir, num_images=20, image_type='defect')
    
    print("\n" + "=" * 50)
    print("Dataset generation complete!")
    print(f"\nTraining images: {80*2} (80 normal + 80 defect)")
    print(f"Validation images: {20*2} (20 normal + 20 defect)")
    print("\nYou can now train the model with:")
    print("python src/train.py --model resnet50 --train_dir data/train --val_dir data/val")

if __name__ == '__main__':
    main()
