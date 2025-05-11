import os
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Base directory
BASE_DIR = Path(__file__).resolve().parents[1]

# Dataset paths
DATASETS = {
    "AffectNet": BASE_DIR / "data" / "Affectnet",
    "RAF-DB": BASE_DIR / "data" / "Rafdb" / "DATASET",
    "FER+": BASE_DIR / "datasets" / "fer-plus"
}

# Class mappings for each dataset
CLASS_MAPPINGS = {
    "AffectNet": {
        "happy": "happiness",
        "sad": "sadness"
    },
    "RAF-DB": {},  # No mapping needed
    "FER+": {}     # No mapping needed
}

# FER+ label mapping (from raw.csv)
FER_LABELS = {
    '0': 'neutral',
    '1': 'happiness',
    '2': 'surprise',
    '3': 'sadness',
    '4': 'anger',
    '5': 'disgust',
    '6': 'fear',
    '7': 'contempt'
}

# RAF-DB class mapping (numeric to emotion)
RAF_LABELS = {
    '1': 'surprise',
    '2': 'fear',
    '3': 'disgust',
    '4': 'happiness',
    '5': 'sadness',
    '6': 'anger',
    '7': 'neutral'
}

def count_rafdb_images(rafdb_dir):
    counts = {v: 0 for v in RAF_LABELS.values()}
    for split in ['train', 'test']:
        split_dir = rafdb_dir / split
        if not split_dir.exists():
            continue
        for class_num, class_name in RAF_LABELS.items():
            class_dir = split_dir / class_num
            if class_dir.exists():
                n_images = len([f for f in class_dir.iterdir() if f.suffix.lower() in ('.jpg', '.jpeg', '.png')])
                counts[class_name] += n_images
    return counts

def count_ferplus_labels(ferplus_dir):
    counts = {v: 0 for v in FER_LABELS.values()}
    processed_csv = ferplus_dir / 'processed.csv'
    if not processed_csv.exists():
        print(f"Warning: processed.csv not found in {ferplus_dir}")
        return counts
    
    df = pd.read_csv(processed_csv)
    print(f"\nFER+ Debug Info:")
    print(f"Total rows in processed.csv: {len(df)}")
    
    # Map column names to emotion names
    column_to_emotion = {
        'neutral': 'neutral',
        'happiness': 'happiness',
        'surprise': 'surprise',
        'sadness': 'sadness',
        'anger': 'anger',
        'disgust': 'disgust',
        'fear': 'fear',
        'contempt': 'contempt'
    }
    
    # Get emotion columns
    emotion_cols = list(column_to_emotion.keys())
    
    # For each row, find the emotion with highest votes
    for _, row in df.iterrows():
        # Get votes for each emotion
        votes = {emotion: row[col] for col, emotion in column_to_emotion.items()}
        # Find emotion with highest votes
        max_emotion = max(votes.items(), key=lambda x: x[1])[0]
        # Increment count for that emotion
        counts[max_emotion] += 1
    
    # Print counts
    for emotion, count in sorted(counts.items(), key=lambda x: x[1], reverse=True):
        print(f"{emotion:10s}: {count:6d} images (highest votes)")
    
    return counts

def count_images_in_directory(directory, class_mapping=None):
    counts = {}
    if not directory.exists():
        print(f"Warning: Directory {directory} does not exist!")
        return counts
    for item in directory.iterdir():
        if item.is_dir():
            class_name = item.name
            if class_mapping and class_name in class_mapping:
                class_name = class_mapping[class_name]
            n_images = len([f for f in item.iterdir() if f.suffix.lower() in ('.jpg', '.jpeg', '.png')])
            counts[class_name] = n_images
    return counts

def plot_distribution(counts, title):
    plt.figure(figsize=(12, 6))
    df = pd.DataFrame({'Emotion': list(counts.keys()), 'Count': list(counts.values())})
    df = df.sort_values('Count', ascending=False)
    sns.barplot(data=df, x='Emotion', y='Count')
    plt.title(title)
    plt.xticks(rotation=45)
    plt.tight_layout()
    for i, count in enumerate(df['Count']):
        plt.text(i, count, str(count), ha='center', va='bottom')
    return plt.gcf()

def main():
    all_results = {}
    for dataset_name, dataset_path in DATASETS.items():
        print(f"\nAnalyzing {dataset_name}...")
        print(f"Path: {dataset_path}")
        if not dataset_path.exists():
            print(f"Error: Dataset path {dataset_path} does not exist!")
            continue
        if dataset_name == "RAF-DB":
            counts = count_rafdb_images(dataset_path)
        elif dataset_name == "FER+":
            counts = count_ferplus_labels(dataset_path)
        else:
            class_mapping = CLASS_MAPPINGS.get(dataset_name, {})
            counts = count_images_in_directory(dataset_path, class_mapping)
        if not counts:
            print(f"Warning: No images/labels found in {dataset_path}")
            continue
        all_results[dataset_name] = counts
        print("\nDistribution:")
        total = 0
        for emotion, count in sorted(counts.items(), key=lambda x: x[1], reverse=True):
            print(f"{emotion:10s}: {count:6d} images")
            total += count
        print(f"{'TOTAL':10s}: {total:6d} images")
        plt.figure(figsize=(12, 6))
        plot_distribution(counts, f"Emotion Distribution - {dataset_name}")
        plt.savefig(f"emotion_dist_{dataset_name.lower()}_original.png")
        plt.close()
    if not all_results:
        print("\nError: No datasets were successfully analyzed!")
        return
    all_emotions = sorted(set().union(*[set(counts.keys()) for counts in all_results.values()]))
    df = pd.DataFrame(index=all_emotions)
    for dataset_name, counts in all_results.items():
        df[dataset_name] = pd.Series(counts)
    df = df.fillna(0)
    df.loc['TOTAL'] = df.sum()
    df.to_csv('original_datasets_comparison.csv')
    print("\nSaved comparison to original_datasets_comparison.csv")
    plt.figure(figsize=(15, 8))
    df.drop('TOTAL').plot(kind='bar')
    plt.title('Emotion Distribution Comparison Across Original Datasets')
    plt.xlabel('Emotion')
    plt.ylabel('Number of Images')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig('original_datasets_comparison.png')
    plt.close()

if __name__ == "__main__":
    main() 