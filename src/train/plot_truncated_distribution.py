from __future__ import annotations
import pathlib
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Config
COMMON_CLASSES = [
    "anger", "contempt", "disgust", "fear",
    "happiness", "neutral", "sadness", "surprise",
]

BASE_DIR = pathlib.Path(__file__).resolve().parents[2]
DATA_DIR = BASE_DIR / "data"
TRUNCATED_DIR = DATA_DIR / "truncated"
FER_DIR = DATA_DIR / "fer-plus_formatted"
RAF_DIR = DATA_DIR / "rafdb_formatted"
AFF_DIR = DATA_DIR / "affectnet_formatted"

def count_images(directory: pathlib.Path) -> dict[str, int]:
    """Numără imaginile pentru fiecare clasă într-un director"""
    counts = {cls: 0 for cls in COMMON_CLASSES}
    for cls in COMMON_CLASSES:
        # Mapăm numele claselor pentru AffectNet
        source_cls = cls
        if "affectnet" in str(directory).lower():
            if cls == "happiness":
                source_cls = "happy"
            elif cls == "sadness":
                source_cls = "sad"
        
        cls_dir = directory/source_cls
        if cls_dir.exists():
            counts[cls] = len([p for p in cls_dir.iterdir() 
                             if p.suffix.lower() in {".jpg", ".jpeg", ".png"}])
    return counts

def plot_distribution(counts: dict[str, int], title: str):
    """Creează un plot cu distribuția emoțiilor"""
    plt.figure(figsize=(12, 6))
    sns.set_style("whitegrid")
    
    # Convertim dicționarul în listă pentru plot
    emotions = list(counts.keys())
    values = list(counts.values())
    
    # Creăm bar plot
    bars = plt.bar(emotions, values, color='skyblue')
    
    # Adăugăm numerele deasupra barelor
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height,
                f'{int(height)}',
                ha='center', va='bottom')
    
    plt.title(title, fontsize=14, pad=20)
    plt.xlabel('Emoție', fontsize=12)
    plt.ylabel('Număr de imagini', fontsize=12)
    plt.xticks(rotation=45)
    plt.tight_layout()
    
    # Salvează plotul
    plt.savefig(f'distribution_{title.lower().replace(" ", "_")}.png', dpi=300, bbox_inches='tight')
    plt.close()

def main():
    # Numără imaginile pentru fiecare set de date
    datasets = {
        "FER+": (FER_DIR, TRUNCATED_DIR/"fer-plus"),
        "RAF-DB": (RAF_DIR, TRUNCATED_DIR/"rafdb"),
        "AffectNet": (AFF_DIR, TRUNCATED_DIR/"affectnet")
    }
    
    print("\n→ Statistici pentru seturile de date originale:")
    original_totals = {}
    for name, (orig_dir, trunc_dir) in datasets.items():
        # Pentru setul original
        train_counts = count_images(orig_dir/"train")
        test_counts = count_images(orig_dir/"test")
        total_counts = {cls: train_counts[cls] + test_counts[cls] 
                       for cls in COMMON_CLASSES}
        original_totals[name] = sum(total_counts.values())
        
        print(f"\n{name} Original Statistics:")
        print("Train:")
        for cls, count in train_counts.items():
            print(f"  {cls}: {count}")
        print("\nTest:")
        for cls, count in test_counts.items():
            print(f"  {cls}: {count}")
        print("\nTotal:")
        for cls, count in total_counts.items():
            print(f"  {cls}: {count}")
        print(f"  Total images: {sum(total_counts.values())}")
    
    print("\n→ Statistici pentru seturile de date truncate:")
    truncated_totals = {}
    for name, (orig_dir, trunc_dir) in datasets.items():
        # Pentru setul truncat
        train_counts = count_images(trunc_dir/"train")
        test_counts = count_images(trunc_dir/"test")
        total_counts = {cls: train_counts[cls] + test_counts[cls] 
                       for cls in COMMON_CLASSES}
        truncated_totals[name] = sum(total_counts.values())
        
        # Plot pentru train
        plot_distribution(train_counts, f"{name} - Train Distribution (Truncated)")
        
        # Plot pentru test
        plot_distribution(test_counts, f"{name} - Test Distribution (Truncated)")
        
        # Plot pentru total
        plot_distribution(total_counts, f"{name} - Total Distribution (Truncated)")
        
        print(f"\n{name} Truncated Statistics:")
        print("Train:")
        for cls, count in train_counts.items():
            print(f"  {cls}: {count}")
        print("\nTest:")
        for cls, count in test_counts.items():
            print(f"  {cls}: {count}")
        print("\nTotal:")
        for cls, count in total_counts.items():
            print(f"  {cls}: {count}")
        print(f"  Total images: {sum(total_counts.values())}")
    
    # Afișează sumele totale
    print("\n→ Sume totale:")
    print("Seturi originale:")
    for name, total in original_totals.items():
        print(f"  {name}: {total} imagini")
    print(f"  Total general original: {sum(original_totals.values())} imagini")
    
    print("\nSeturi truncate:")
    for name, total in truncated_totals.items():
        print(f"  {name}: {total} imagini")
    print(f"  Total general truncat: {sum(truncated_totals.values())} imagini")
    
    # Calculează și afișează reducerea
    total_original = sum(original_totals.values())
    total_truncated = sum(truncated_totals.values())
    reduction = ((total_original - total_truncated) / total_original) * 100
    print(f"\nReducere totală: {reduction:.2f}%")

if __name__ == "__main__":
    main() 