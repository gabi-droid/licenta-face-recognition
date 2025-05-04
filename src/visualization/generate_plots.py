import os
import pandas as pd
from plots import plot_fer_distribution, plot_raf_distribution, plot_affectnet_distribution, plot_combined_distribution

def count_images_in_directory(directory):
    """
    Numără imaginile din fiecare subdirector și creează un DataFrame
    """
    data = []
    for emotion in os.listdir(directory):
        emotion_dir = os.path.join(directory, emotion)
        if os.path.isdir(emotion_dir):
            num_images = len([f for f in os.listdir(emotion_dir) if os.path.isfile(os.path.join(emotion_dir, f))])
            data.extend([{'emotion_name': emotion}] * num_images)
    return pd.DataFrame(data)

def main():
    # Încărcăm datele pentru FER+
    print("Încărcăm datele FER+...")
    fer_train_dir = 'data/fer-plus_formatted/train'
    fer_val_dir = 'data/fer-plus_formatted/val'
    fer_test_dir = 'data/fer-plus_formatted/test'
    
    df_fer_train = count_images_in_directory(fer_train_dir)
    df_fer_val = count_images_in_directory(fer_val_dir)
    df_fer_test = count_images_in_directory(fer_test_dir)
    df_fer = pd.concat([df_fer_train, df_fer_val, df_fer_test], ignore_index=True)
    
    # Încărcăm datele pentru RAF-DB
    print("\nÎncărcăm datele RAF-DB...")
    raf_train_dir = 'data/rafdb_formatted/train'
    raf_test_dir = 'data/rafdb_formatted/test'
    
    df_raf_train = count_images_in_directory(raf_train_dir)
    df_raf_test = count_images_in_directory(raf_test_dir)
    df_raf = pd.concat([df_raf_train, df_raf_test], ignore_index=True)
    
    # Încărcăm datele pentru AffectNet
    print("\nÎncărcăm datele AffectNet...")
    affectnet_dir = 'data/affectnet_formatted'
    df_affectnet = count_images_in_directory(affectnet_dir)
    
    # Standardizăm numele emoțiilor pentru AffectNet
    emotion_mapping = {
        'happy': 'happiness',
        'sad': 'sadness'
    }
    df_affectnet['emotion_name'] = df_affectnet['emotion_name'].map(lambda x: emotion_mapping.get(x, x))
    
    # Generăm graficele
    print("\nGenerăm graficele...")
    plot_fer_distribution(df_fer)
    plot_raf_distribution(df_raf)
    plot_affectnet_distribution(df_affectnet)
    plot_combined_distribution(df_fer, df_raf, df_affectnet)
    
    print("\nGraficele au fost salvate în directorul curent.")

if __name__ == '__main__':
    main() 