import os
import numpy as np
import pandas as pd
from PIL import Image
from tqdm import tqdm

def convert_rafdb_to_rgb():
    # Create output directory
    output_dir = 'data/rafdb_formatted'
    os.makedirs(output_dir, exist_ok=True)
    
    # Process images from RAF-DB directory
    rafdb_dir = 'data/Rafdb/DATASET'
    emotion_map = {
        '1': 'surprise',
        '2': 'fear',
        '3': 'disgust',
        '4': 'happiness',
        '5': 'sadness',
        '6': 'anger',
        '7': 'neutral'
    }
    
    # Process both train and test sets
    for dataset_type in ['train', 'test']:
        print(f"\nProcessing {dataset_type} set...")
        
        # Create output subdirectory
        output_subdir = os.path.join(output_dir, dataset_type)
        os.makedirs(output_subdir, exist_ok=True)
        
        # Process each emotion directory
        for emotion_num, emotion_name in emotion_map.items():
            print(f"\nProcessing {emotion_name}...")
            
            # Create emotion subdirectory in output
            output_emotion_dir = os.path.join(output_subdir, emotion_name)
            os.makedirs(output_emotion_dir, exist_ok=True)
            
            # Get all image files
            emotion_dir = os.path.join(rafdb_dir, dataset_type, emotion_num)
            image_files = [f for f in os.listdir(emotion_dir) 
                          if f.lower().endswith(('.jpg', '.png'))]
            total_files = len(image_files)
            
            print(f"Found {total_files} images in {emotion_name}")
            
            # Process all images in the emotion directory
            for img_name in tqdm(image_files, desc=f'Processing {emotion_name}'):
                try:
                    # Read the image
                    img_path = os.path.join(emotion_dir, img_name)
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
    convert_rafdb_to_rgb() 