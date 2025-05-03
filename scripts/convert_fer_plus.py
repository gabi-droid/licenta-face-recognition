import os
import numpy as np
import pandas as pd
from PIL import Image
from tqdm import tqdm

def convert_fer_plus_to_rgb():
    # Read the raw CSV file
    df_raw = pd.read_csv('datasets/fer-plus/raw.csv')
    df_processed = pd.read_csv('datasets/fer-plus/processed.csv')
    
    # Create output directories if they don't exist
    output_dirs = {
        'Training': 'data/fer-plus_formatted/train',
        'PublicTest': 'data/fer-plus_formatted/val',
        'PrivateTest': 'data/fer-plus_formatted/test'
    }
    
    for dir_path in output_dirs.values():
        os.makedirs(dir_path, exist_ok=True)
    
    # Process all images
    for idx, (_, row) in enumerate(tqdm(df_processed.iterrows(), total=len(df_processed))):
        usage = row['Usage']
        
        # Get the emotion with highest score
        emotions = ['neutral', 'happiness', 'surprise', 'sadness', 'anger', 'disgust', 'fear', 'contempt', 'unknown']
        emotion_scores = [row[emotion] for emotion in emotions]
        emotion_idx = np.argmax(emotion_scores)
        emotion = emotions[emotion_idx]
        
        # Create emotion subdirectory
        emotion_dir = os.path.join(output_dirs[usage], emotion)
        os.makedirs(emotion_dir, exist_ok=True)
        
        # Get the pixel values from raw CSV
        pixels_str = df_raw.iloc[idx]['pixels']
        pixels = np.array([int(p) for p in pixels_str.split()], dtype=np.uint8)
        img_array = pixels.reshape(48, 48)
        
        # Convert numpy array to PIL Image
        img = Image.fromarray(img_array, mode='L')  # 'L' mode for grayscale
        
        # Resize to 224x224
        img_resized = img.resize((224, 224), Image.Resampling.LANCZOS)
        
        # Convert to RGB
        img_rgb = img_resized.convert('RGB')
        
        # Convert back to numpy array for debugging
        img_array_final = np.array(img_rgb)
        print(f"Image shape after RGB conversion: {img_array_final.shape}")
        print(f"Image dtype: {img_array_final.dtype}")
        print(f"Image min value: {img_array_final.min()}, max value: {img_array_final.max()}")
        
        # Generate image filename
        image_name = f'image_{idx:06d}.png'
        
        # Save the new image
        output_path = os.path.join(emotion_dir, image_name)
        img_rgb.save(output_path, format='PNG', quality=100)
        print(f"Saved image to: {output_path}")

if __name__ == '__main__':
    convert_fer_plus_to_rgb() 