"""
Module for loading and preprocessing RAF-DB dataset
"""

import pandas as pd

def load_raf_db_data(train_labels_path, test_labels_path):
    """
    Încarcă datele din setul RAF-DB
    Args:
        train_labels_path: calea către fișierul cu etichete pentru setul de antrenare
        test_labels_path: calea către fișierul cu etichete pentru setul de test
    Returns:
        DataFrame cu datele procesate
    """
    # Încărcăm datele de antrenare și test
    train_df = pd.read_csv(train_labels_path)
    test_df = pd.read_csv(test_labels_path)
    
    # Adăugăm o coloană pentru a indica setul (train/test)
    train_df['Usage'] = 'Training'
    test_df['Usage'] = 'Test'
    
    # Combinăm datele
    df = pd.concat([train_df, test_df], ignore_index=True)
    
    # Mapăm etichetele la numele emoțiilor (RAF-DB folosește 1-7 pentru etichete)
    emotion_map = {
        1: 'surprise',
        2: 'fear',
        3: 'disgust',
        4: 'happiness',
        5: 'sadness',
        6: 'anger',
        7: 'neutral'
    }
    
    # Convertim etichetele numerice în nume de emoții
    df['emotion_name'] = df['label'].map(emotion_map)
    
    # Creăm coloane one-hot pentru fiecare emoție
    emotion_columns = list(emotion_map.values())
    for emotion in emotion_columns:
        df[emotion] = (df['emotion_name'] == emotion).astype(int)
    
    # Adăugăm coloane pentru totalul voturilor și procentaje
    df['total_votes'] = 1
    for emotion in emotion_columns:
        df[f'{emotion}_percent'] = df[emotion] * 100
    
    # Adăugăm coloana pentru sursa datelor
    df['Source'] = 'RAF-DB'
    
    return df 