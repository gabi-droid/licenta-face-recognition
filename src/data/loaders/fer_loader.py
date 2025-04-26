"""
Module for loading and preprocessing FER+ dataset
"""

import pandas as pd

def load_fer_data(path_to_fer):
    """
    Incarca datele din fer2013new.csv și face procesarea inițială
    Filtrează doar imaginile care au cel puțin 7 voturi pentru aceeași emoție
    Args:
        path_to_fer: calea catre fișierul fer2013new.csv
    Returns:
        DataFrame cu datele procesate și filtrate
    """
    df = pd.read_csv(path_to_fer)
    
    emotion_columns = ['neutral', 'happiness', 'surprise', 'sadness', 
                      'anger', 'disgust', 'fear', 'contempt']
    
    # Calculăm numărul total de voturi pentru fiecare imagine
    df['total_votes'] = df[emotion_columns + ['unknown', 'NF']].sum(axis=1)
    
    # Calculăm procentajele pentru fiecare emoție
    for col in emotion_columns:
        df[f'{col}_percent'] = df[col] / df['total_votes'] * 100
    
    # Găsim emoția dominantă pentru fiecare imagine
    df['dominant_emotion'] = df[emotion_columns].idxmax(axis=1)
    df['max_votes'] = df[emotion_columns].max(axis=1)
    
    # Filtrăm imaginile care au cel puțin 7 voturi pentru emoția dominantă
    df_filtered = df[df['max_votes'] >= 7].copy()
    
    # Adăugăm coloana pentru sursa datelor
    df_filtered['Source'] = 'FER+'
    
    # Redenumim coloana pentru consistență cu RAF-DB
    df_filtered['emotion_name'] = df_filtered['dominant_emotion']

    return df_filtered

def get_emotion_columns():
    """
    Returnează lista de coloane pentru emoții (fără usage, NF, unknown)
    """
    return ['neutral', 'happiness', 'surprise', 'sadness', 
            'anger', 'disgust', 'fear', 'contempt'] 