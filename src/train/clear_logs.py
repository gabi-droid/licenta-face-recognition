import shutil
from pathlib import Path

# Configurare
BASE_DIR = Path(__file__).resolve().parents[2]
LOGS_DIR = BASE_DIR / "logs"

def clear_logs():
    """Șterge toate logurile din TensorBoard"""
    if LOGS_DIR.exists():
        print(f"Șterg logurile din {LOGS_DIR}...")
        shutil.rmtree(LOGS_DIR)
        print("Gata!")
    else:
        print("Nu există loguri de șters.")

if __name__ == "__main__":
    clear_logs() 