"""
Module for loading and preprocessing AffectNet dataset
"""

import pandas as pd

def load_affectnet_data(labels_path):
    """
    Încarcă datele din setul AffectNet
    Args:
        labels_path: calea către fișierul cu etichete AffectNet
    Returns:
        DataFrame cu datele procesate
    """
    # Încărcăm datele
    df = pd.read_csv(labels_path)
    
    # Redenumim coloana pentru consistență cu celelalte seturi de date
    df['emotion_name'] = df['label']
    
    # Lista emoțiilor disponibile
    emotion_columns = df['emotion_name'].unique().tolist()
    
    # Creăm coloane one-hot pentru fiecare emoție
    for emotion in emotion_columns:
        df[emotion] = (df['emotion_name'] == emotion).astype(int)
    
    # Adăugăm coloane pentru totalul voturilor și procentaje
    df['total_votes'] = 1
    for emotion in emotion_columns:
        df[f'{emotion}_percent'] = df[emotion] * 100
    
    # Adăugăm coloana pentru sursa datelor
    df['Source'] = 'AffectNet'
    
    return df 