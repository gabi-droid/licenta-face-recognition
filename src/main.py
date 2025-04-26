"""
Main entry point for the face recognition project
"""

import os
from data.loaders import load_fer_data, load_raf_db_data
from visualization.plots import (
    plot_fer_distribution,
    plot_raf_distribution,
    plot_combined_distribution
)

def main():
    """
    Punctul principal de intrare pentru procesarea și analiza datelor
    """
    base_path = os.path.dirname(os.path.abspath(__file__))
    parent_path = os.path.dirname(base_path)
    
    # Căile către fișiere
    fer_path = os.path.join(parent_path, "datasets", "fer-plus", "processed.csv")
    raf_train_path = os.path.join(parent_path, "datasets", "raf-db", "train_labels.csv")
    raf_test_path = os.path.join(parent_path, "datasets", "raf-db", "test_labels.csv")
    
    try:
        # Încărcăm datele FER+
        print("Încărcăm datele FER+...")
        df_fer = load_fer_data(fer_path)
        
        # Încărcăm datele RAF-DB
        print("\nÎncărcăm datele RAF-DB...")
        df_raf = load_raf_db_data(raf_train_path, raf_test_path)
        
        # Generăm vizualizările
        print("\nGenerăm vizualizările...")
        plot_fer_distribution(df_fer)
        plot_raf_distribution(df_raf)
        plot_combined_distribution(df_fer, df_raf)
        
        print("\nProcesare completă! Vizualizările au fost salvate în directorul curent.")
        
    except FileNotFoundError as e:
        print(f"Eroare: Nu s-a putut găsi fișierul: {str(e)}")
    except Exception as e:
        print(f"A apărut o eroare: {str(e)}")

if __name__ == "__main__":
    main() 