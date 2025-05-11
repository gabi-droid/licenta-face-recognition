import os
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from collections import Counter
import seaborn as sns

# Configurare
BASE_DIR = Path(__file__).resolve().parents[2]
DATA_DIR = BASE_DIR / "data"
FORMATTED_DIRS = {
    "FER+": DATA_DIR / "fer-plus_formatted",
    "RAF": DATA_DIR / "rafdb_formatted",
    "AffectNet": DATA_DIR / "affectnet_formatted"
}

COMMON_CLASSES = [
    "anger", "disgust", "fear",
    "happiness", "neutral", "sadness", "surprise"
]

SPLITS = ["train", "val", "test"]

def analyze_dataset(dataset_path: Path, dataset_name: str):
    """Analizează distribuția claselor într-un dataset"""
    print(f"\nAnalizăm {dataset_name}...")
    
    # Verificăm dacă directorul există
    if not dataset_path.exists():
        print(f"  ⚠️  Directorul {dataset_path} nu există!")
        return None
    
    # Analizăm fiecare split
    split_counts = {}
    for split in SPLITS:
        split_path = dataset_path / split
        if not split_path.exists():
            print(f"  ⚠️  Split-ul {split} nu există în {dataset_name}")
            continue
            
        # Numărăm imaginile per clasă
        class_counts = Counter()
        total_images = 0
        
        for class_name in COMMON_CLASSES:
            class_dir = split_path / class_name
            if class_dir.exists():
                n_images = len(list(class_dir.glob("*.jpg")))
                class_counts[class_name] = n_images
                total_images += n_images
            else:
                print(f"  ⚠️  Directorul {class_name} nu există în {dataset_name}/{split}")
        
        if total_images > 0:
            split_counts[split] = class_counts
            print(f"\nStatistici {dataset_name}/{split}:")
            print(f"  Total imagini: {total_images}")
            print("\nDistribuția claselor:")
            for class_name in COMMON_CLASSES:
                count = class_counts[class_name]
                percentage = (count / total_images) * 100
                print(f"  {class_name:10s}: {count:5d} imagini ({percentage:5.1f}%)")
    
    return split_counts

def plot_distributions(all_counts):
    """Generează un plot pentru distribuțiile claselor"""
    # Setăm stilul
    sns.set_style("whitegrid")
    plt.rcParams.update({'font.size': 12})
    
    # Creăm subplot-uri pentru fiecare split
    fig, axes = plt.subplots(len(SPLITS), 2, figsize=(15, 5*len(SPLITS)))
    
    for i, split in enumerate(SPLITS):
        ax1, ax2 = axes[i]
        
        # Plot 1: Bar plot pentru numărul absolut de imagini
        x = np.arange(len(COMMON_CLASSES))
        width = 0.25
        
        for j, (dataset_name, split_counts) in enumerate(all_counts.items()):
            if split in split_counts:
                values = [split_counts[split][cls] for cls in COMMON_CLASSES]
                ax1.bar(x + j*width, values, width, label=dataset_name)
        
        ax1.set_ylabel('Număr de imagini')
        ax1.set_title(f'Distribuția claselor în {split} set')
        ax1.set_xticks(x + width)
        ax1.set_xticklabels(COMMON_CLASSES, rotation=45)
        ax1.legend()
        
        # Plot 2: Line plot pentru procente
        for dataset_name, split_counts in all_counts.items():
            if split in split_counts:
                counts = split_counts[split]
                total = sum(counts.values())
                percentages = [counts[cls]/total * 100 for cls in COMMON_CLASSES]
                ax2.plot(COMMON_CLASSES, percentages, marker='o', label=dataset_name)
        
        ax2.set_ylabel('Procent (%)')
        ax2.set_title(f'Distribuția procentuală în {split} set')
        ax2.set_xticklabels(COMMON_CLASSES, rotation=45)
        ax2.legend()
    
    plt.tight_layout()
    plt.savefig('class_distributions.png')
    print("\nPlot salvat în 'class_distributions.png'")

def main():
    print("Analizăm distribuția claselor în dataset-urile formatate...")
    
    all_counts = {}
    for dataset_name, dataset_path in FORMATTED_DIRS.items():
        split_counts = analyze_dataset(dataset_path, dataset_name)
        if split_counts:
            all_counts[dataset_name] = split_counts
    
    if all_counts:
        plot_distributions(all_counts)
    else:
        print("\n⚠️  Nu s-au găsit date pentru analiză!")

if __name__ == "__main__":
    main() 