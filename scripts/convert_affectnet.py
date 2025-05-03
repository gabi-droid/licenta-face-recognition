import os
import numpy as np
import pandas as pd
from PIL import Image
from tqdm import tqdm

def convert_affectnet_to_rgb():
    # Create output directory
    output_dir = 'data/affectnet_formatted'
    os.makedirs(output_dir, exist_ok=True)
    
    # Process images from AffectNet directory
    affectnet_dir = 'data/AffectNet'
    emotion_dirs = ['anger', 'contempt', 'disgust', 'fear', 'happy', 'neutral', 'sad', 'surprise']
    total_dirs = len(emotion_dirs)
    
    print(f"\nProcessing images from {total_dirs} emotion directories...")
    
    for idx, emotion_dir in enumerate(emotion_dirs):
        print(f"\nProcessing {emotion_dir} ({idx+1}/{total_dirs})")
        
        # Create emotion subdirectory in output
        output_emotion_dir = os.path.join(output_dir, emotion_dir)
        os.makedirs(output_emotion_dir, exist_ok=True)
        
        # Get all image files (both jpg and png)
        image_files = [f for f in os.listdir(os.path.join(affectnet_dir, emotion_dir)) 
                      if f.lower().endswith(('.jpg', '.png'))]
        total_files = len(image_files)
        
        print(f"Found {total_files} images in {emotion_dir}")
        
        # Process all images in the emotion directory
        for img_name in tqdm(image_files, desc=f'Processing {emotion_dir}'):
            try:
                # Read the image
                img_path = os.path.join(affectnet_dir, emotion_dir, img_name)
                img = Image.open(img_path)
                
                # Convert to RGB if not already
                if img.mode != 'RGB':
                    img = img.convert('RGB')
                
                # Resize to 224x224
                img_resized = img.resize((224, 224), Image.Resampling.LANCZOS)
                
                # Save the new image
                output_path = os.path.join(output_emotion_dir, img_name)
                img_resized.save(output_path, format='PNG', quality=100)
            except Exception as e:
                print(f"Error processing {img_name}: {str(e)}")
                continue

if __name__ == '__main__':
    convert_affectnet_to_rgb() 