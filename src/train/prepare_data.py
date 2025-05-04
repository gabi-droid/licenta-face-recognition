from __future__ import annotations
import os, shutil, random, pathlib
from typing import Dict, List
import numpy as np
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from PIL import Image
from tqdm import tqdm

# Config
COMMON_CLASSES: List[str] = [
    "anger", "contempt", "disgust", "fear",
    "happiness", "neutral", "sadness", "surprise",
]
IMG_SIZE, BATCH_SIZE = (224, 224), 128
TRAIN_LIMIT = 1000
TEST_LIMIT = 200

BASE_DIR = pathlib.Path(__file__).resolve().parents[2]
DATA_DIR = BASE_DIR / "data"
FER_DIR = DATA_DIR / "fer-plus_formatted"
RAF_DIR = DATA_DIR / "rafdb_formatted"
AFF_DIR = DATA_DIR / "affectnet_formatted"

# Create new directory for truncated data
TRUNCATED_DIR = DATA_DIR / "truncated"
TRUNCATED_DIR.mkdir(exist_ok=True)

def split_affectnet(root: pathlib.Path, test_ratio: float = 0.20):
    tr, ts = root/"train", root/"test"
    if tr.exists() and ts.exists():
        return tr, ts
    print("→ Împărţim AffectNet…")
    tr.mkdir(parents=True, exist_ok=True); ts.mkdir(exist_ok=True)
    rng = random.Random(42)
    for cls in sorted(os.listdir(root)):
        src = root/cls
        if not src.is_dir() or cls in {"train", "test"}: continue
        (tr/cls).mkdir(exist_ok=True); (ts/cls).mkdir(exist_ok=True)
        imgs = [p for p in src.iterdir() if p.suffix.lower() in {".jpg", ".jpeg", ".png"}]
        rng.shuffle(imgs)
        cut = int(len(imgs)*test_ratio)
        for i,p in enumerate(imgs):
            dst = ts/cls/p.name if i<cut else tr/cls/p.name
            if not dst.exists(): shutil.copy2(p,dst)
    return tr, ts

def prepare_dataset(
    source_dir: pathlib.Path,
    target_dir: pathlib.Path,
    is_train: bool = True,
    limit: int = TRAIN_LIMIT
) -> None:
    """Pregătește un set de date prin selectarea unui număr limitat de imagini"""
    print(f"\n→ Pregătire {source_dir.name} {'train' if is_train else 'test'} set")
    
    # Creează structura de directoare
    for cls in COMMON_CLASSES:
        (target_dir/cls).mkdir(parents=True, exist_ok=True)
    
    # Procesează fiecare clasă
    for cls in tqdm(COMMON_CLASSES, desc="Procesare clase"):
        # Mapăm numele claselor pentru AffectNet
        source_cls = cls
        if "affectnet" in str(source_dir).lower():
            if cls == "happiness":
                source_cls = "happy"
            elif cls == "sadness":
                source_cls = "sad"
        
        source_cls_dir = source_dir/source_cls
        if not source_cls_dir.exists():
            print(f"  ⚠️ Clasa {source_cls} nu există în {source_dir.name}")
            continue
        
        # Obține lista de imagini
        images = [p for p in source_cls_dir.iterdir() 
                 if p.suffix.lower() in {".jpg", ".jpeg", ".png"}]
        
        # Selectează un subset aleator
        if len(images) > limit:
            images = random.sample(images, limit)
        
        # Copiază imaginile
        count = 0
        for img_path in tqdm(images, desc=f"  Copiere {cls}", leave=False):
            try:
                # Verifică și convertește imaginea
                with Image.open(img_path) as img:
                    if img.mode != 'RGB':
                        img = img.convert('RGB')
                    
                    # Salvează imaginea
                    target_path = target_dir/cls/f"{cls}_{count:04d}.jpg"
                    img.save(target_path, "JPEG", quality=95)
                    count += 1
            except Exception as e:
                print(f"  ⚠️ Eroare la procesarea {img_path.name}: {e}")
        
        print(f"  ✅ {cls}: {count} imagini copiate")

def main():
    # Șterge directorul truncat dacă există
    if TRUNCATED_DIR.exists():
        shutil.rmtree(TRUNCATED_DIR)
    
    # Pregătește seturile de date
    prepare_dataset(
        DATA_DIR/"fer-plus_formatted"/"train",
        TRUNCATED_DIR/"fer-plus"/"train",
        is_train=True,
        limit=TRAIN_LIMIT
    )
    prepare_dataset(
        DATA_DIR/"fer-plus_formatted"/"test",
        TRUNCATED_DIR/"fer-plus"/"test",
        is_train=False,
        limit=TEST_LIMIT
    )
    
    prepare_dataset(
        DATA_DIR/"rafdb_formatted"/"train",
        TRUNCATED_DIR/"rafdb"/"train",
        is_train=True,
        limit=TRAIN_LIMIT
    )
    prepare_dataset(
        DATA_DIR/"rafdb_formatted"/"test",
        TRUNCATED_DIR/"rafdb"/"test",
        is_train=False,
        limit=TEST_LIMIT
    )
    
    prepare_dataset(
        DATA_DIR/"affectnet_formatted"/"train",
        TRUNCATED_DIR/"affectnet"/"train",
        is_train=True,
        limit=TRAIN_LIMIT
    )
    prepare_dataset(
        DATA_DIR/"affectnet_formatted"/"test",
        TRUNCATED_DIR/"affectnet"/"test",
        is_train=False,
        limit=TEST_LIMIT
    )

if __name__ == "__main__":
    main() 