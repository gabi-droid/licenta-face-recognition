import pandas as pd
import matplotlib.pyplot as plt
from collections import Counter
from pathlib import Path
from scipy.spatial.distance import jensenshannon

# Configurare
DATASETS = ["fer-plus", "rafdb", "affectnet"]
SPLITS   = ["train", "val", "test"]
CLASSES  = ["anger", "contempt", "disgust", "fear", "happiness", "neutral", "sadness", "surprise"]

def count_imgs(root):
    """Numără imaginile per clasă într-un director"""
    counts = Counter()
    for cls in CLASSES:
        counts[cls] += len(list((root/cls).glob("*.jpg")))
    return counts

def analyze_distributions():
    """Analizează distribuțiile de imagini"""
    # Colectăm datele
    rows = []
    for ds in DATASETS:
        for sp in SPLITS:
            root = Path("data/truncated")/ds/sp
            if not root.exists():
                print(f"Warning: {root} nu există")
                continue
                
            cnt = count_imgs(root)
            total = sum(cnt.values())
            for emo in CLASSES:
                rows.append({
                    "dataset": ds, "split": sp, "emotion": emo,
                    "count": cnt[emo], "pct": cnt[emo]/total if total else 0
                })

    # Creăm DataFrame-ul
    df = pd.DataFrame(rows)
    
    # 1. Distribuția per dataset și split (procente)
    print("\n1. Distribuția per dataset și split (procente):")
    pivot = df.pivot_table(index=["dataset", "split"], columns="emotion",
                          values="pct", fill_value=0)
    print(pivot.round(3))
    
    # 1b. Distribuția per dataset și split (numere brute)
    print("\n1b. Distribuția per dataset și split (numere brute):")
    pivot_counts = df.pivot_table(index=["dataset", "split"], columns="emotion",
                                values="count", fill_value=0)
    print(pivot_counts)
    
    # 2. Distribuția globală train vs val vs test
    print("\n2. Distribuția globală train vs val vs test:")
    global_dist = df.groupby(["split", "emotion"])["count"].sum()
    global_dist = global_dist.unstack(level=0)
    
    # Calculăm procentele pentru fiecare split
    for split in SPLITS:
        if split in global_dist.columns:
            total = global_dist[split].sum()
            global_dist[f"{split}_pct"] = global_dist[split] / total
    
    # Afișăm procentele
    pct_cols = [f"{split}_pct" for split in SPLITS if f"{split}_pct" in global_dist.columns]
    print(global_dist[pct_cols].round(3))
    
    # Calculăm divergența Jensen-Shannon între distribuții
    if "train_pct" in global_dist.columns and "val_pct" in global_dist.columns:
        js_train_val = jensenshannon(global_dist["train_pct"], global_dist["val_pct"])
        print(f"\nJensen-Shannon distance train-val: {js_train_val:.3f}")
    
    if "train_pct" in global_dist.columns and "test_pct" in global_dist.columns:
        js_train_test = jensenshannon(global_dist["train_pct"], global_dist["test_pct"])
        print(f"Jensen-Shannon distance train-test: {js_train_test:.3f}")
    
    # 3. Plot distribuția globală
    plt.figure(figsize=(12, 6))
    global_dist[pct_cols].plot(kind="bar")
    plt.title("Distribuția globală train vs val vs test")
    plt.xlabel("Emoție")
    plt.ylabel("Procent")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig("distribution_analysis.png")
    print("\nPlot salvat în distribution_analysis.png")

if __name__ == "__main__":
    analyze_distributions() 