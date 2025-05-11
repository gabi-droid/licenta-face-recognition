import os
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Base directory
BASE_DIR = Path(__file__).resolve().parents[1]

# Dataset paths
DATASETS = {
    "AffectNet": BASE_DIR / "data" / "truncated" / "affectnet",
    "RAF-DB": BASE_DIR / "data" / "truncated" / "rafdb",
    "FER+": BASE_DIR / "data" / "truncated" / "fer-plus"
}

# Common emotion classes across all datasets
COMMON_CLASSES = [
    "anger", "disgust", "fear", "contempt",
    "happiness", "neutral", "sadness", "surprise"
]

def count_images_in_split(dataset_path, split):
    """Count images in a specific split (train/val/test) of a dataset"""
    counts = {emotion: 0 for emotion in COMMON_CLASSES}
    split_dir = dataset_path / split
    if not split_dir.exists():
        print(f"Warning: Split directory {split_dir} does not exist!")
        return counts
    
    for emotion in COMMON_CLASSES:
        emotion_dir = split_dir / emotion
        if emotion_dir.exists():
            n_images = len([f for f in emotion_dir.iterdir() if f.suffix.lower() in ('.jpg', '.jpeg', '.png')])
            counts[emotion] = n_images
    return counts

def analyze_dataset(dataset_path, dataset_name):
    """Analyze a single dataset across all splits"""
    print(f"\nAnalyzing {dataset_name}...")
    print(f"Path: {dataset_path}")
    
    if not dataset_path.exists():
        print(f"Error: Dataset path {dataset_path} does not exist!")
        return None
    
    # Count images in each split
    splits = ['train', 'val', 'test']
    split_counts = {}
    total_counts = {emotion: 0 for emotion in COMMON_CLASSES}
    
    for split in splits:
        counts = count_images_in_split(dataset_path, split)
        split_counts[split] = counts
        for emotion, count in counts.items():
            total_counts[emotion] += count
    
    # Print detailed statistics
    print("\nDistribution by split:")
    for split in splits:
        print(f"\n{split.upper()} split:")
        for emotion, count in sorted(split_counts[split].items(), key=lambda x: x[1], reverse=True):
            print(f"{emotion:10s}: {count:6d} images")
    
    print("\nTotal distribution:")
    for emotion, count in sorted(total_counts.items(), key=lambda x: x[1], reverse=True):
        print(f"{emotion:10s}: {count:6d} images")
    
    # Create visualization
    plt.figure(figsize=(15, 8))
    
    # Create DataFrame for plotting
    df = pd.DataFrame(split_counts).T
    df = df[COMMON_CLASSES]  # Ensure consistent order
    
    # Plot
    ax = df.plot(kind='bar', stacked=True)
    plt.title(f'Emotion Distribution - {dataset_name} (Truncated)')
    plt.xlabel('Split')
    plt.ylabel('Number of Images')
    plt.legend(title='Emotion', bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    
    # Add value labels
    for c in ax.containers:
        ax.bar_label(c, label_type='center')
    
    # Save plot
    plt.savefig(f"emotion_dist_{dataset_name.lower()}_truncated.png", bbox_inches='tight')
    plt.close()
    
    return total_counts

def main():
    all_results = {}
    
    # Analyze each dataset
    for dataset_name, dataset_path in DATASETS.items():
        counts = analyze_dataset(dataset_path, dataset_name)
        if counts:
            all_results[dataset_name] = counts
    
    if not all_results:
        print("\nError: No datasets were successfully analyzed!")
        return
    
    # Create comparison DataFrame
    df = pd.DataFrame(all_results)
    df = df.fillna(0)
    df.loc['TOTAL'] = df.sum()
    
    # Save comparison to CSV
    df.to_csv('truncated_datasets_comparison.csv')
    print("\nSaved comparison to truncated_datasets_comparison.csv")
    
    # Create comparison plot
    plt.figure(figsize=(15, 8))
    df.drop('TOTAL').plot(kind='bar')
    plt.title('Emotion Distribution Comparison Across Truncated Datasets')
    plt.xlabel('Emotion')
    plt.ylabel('Number of Images')
    plt.xticks(rotation=45)
    plt.legend(title='Dataset', bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    
    # Add value labels
    for c in plt.gca().containers:
        plt.gca().bar_label(c, label_type='edge')
    
    plt.savefig('truncated_datasets_comparison.png', bbox_inches='tight')
    plt.close()

if __name__ == "__main__":
    main() 