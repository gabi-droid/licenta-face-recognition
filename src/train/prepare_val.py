import os
import shutil
import random
from pathlib import Path
import logging

# Configurare logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Configurare
BASE_DIR = Path(__file__).resolve().parents[2]
DATA_DIR = BASE_DIR / "data"
TRUNCATED_DIR = DATA_DIR / "truncated"

# Emoții comune
COMMON_CLASSES = [
    "anger", "contempt", "disgust", "fear",
    "happiness", "neutral", "sadness", "surprise"
]

# Seturi de date
DATASETS = ["fer-plus", "rafdb", "affectnet"]

def prepare_dataset(dataset_name: str):
    """Pregătește setul de date cu split 80/10/10 pentru train/val/test"""
    logger.info(f"Pregătim {dataset_name}...")
    
    # Directoare sursă și destinație
    src_dir = TRUNCATED_DIR / dataset_name / "train"
    val_dir = TRUNCATED_DIR / dataset_name / "val"
    test_dir = TRUNCATED_DIR / dataset_name / "test"
    
    # Ștergem directoarele val și test dacă există
    if val_dir.exists():
        shutil.rmtree(val_dir)
    if test_dir.exists():
        shutil.rmtree(test_dir)
    
    # Creăm directoarele
    val_dir.mkdir(parents=True)
    test_dir.mkdir(parents=True)
    
    # Pentru fiecare emoție
    for emotion in COMMON_CLASSES:
        logger.info(f"  Procesăm {emotion}...")
        
        # Creăm directoarele pentru emoție
        (val_dir / emotion).mkdir(exist_ok=True)
        (test_dir / emotion).mkdir(exist_ok=True)
        
        # Obținem toate imaginile pentru această emoție
        emotion_dir = src_dir / emotion
        if not emotion_dir.exists():
            logger.warning(f"    Directorul {emotion_dir} nu există, sărim peste...")
            continue
            
        images = []
        for ext in [".jpg", ".jpeg", ".png"]:
            images.extend(list(emotion_dir.glob(f"*{ext}")))
        
        if not images:
            logger.warning(f"    Nu s-au găsit imagini în {emotion_dir}, sărim peste...")
            continue
        
        # Amestecăm imaginile
        random.shuffle(images)
        
        # Calculăm indecșii pentru split
        n_images = len(images)
        val_start = int(n_images * 0.8)  # 80% pentru train
        test_start = int(n_images * 0.9)  # 90% pentru train+val
        
        # Copiem imaginile în seturile val și test
        for img in images[val_start:test_start]:
            shutil.copy2(img, val_dir / emotion / img.name)
        for img in images[test_start:]:
            shutil.copy2(img, test_dir / emotion / img.name)
        
        # Ștergem imaginile din train care au fost mutate în val și test
        for img in images[val_start:]:
            img.unlink()
        
        logger.info(f"    {emotion}: {n_images} imagini totale")
        logger.info(f"      Train: {val_start} imagini")
        logger.info(f"      Val: {test_start - val_start} imagini")
        logger.info(f"      Test: {n_images - test_start} imagini")

def main():
    """Funcția principală"""
    logger.info("Începem pregătirea seturilor de date...")
    
    # Pregătim fiecare set de date
    for dataset in DATASETS:
        prepare_dataset(dataset)
    
    logger.info("Gata!")

if __name__ == "__main__":
    main()