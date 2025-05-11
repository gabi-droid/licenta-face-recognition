# ============================================================================
#  Patt‑Lite – train_patt_lite.py  ✧ versiune optimizată (fix fine‑tune layer)
# ----------------------------------------------------------------------------
#  • Corectează numele backbone‑ului MobileNet → "mobilenet_backbone".
#  • Fine‑tuning-ul se face acum pe ultimele 40 de straturi ale acestui backbone.
# ============================================================================

from __future__ import annotations
import os, shutil, datetime, random, pathlib
from typing import Tuple, List, Dict
from pathlib import Path

import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications.mobilenet import preprocess_input
import logging

# Configurare logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ----------------------------------------------------------------------------
# CONFIG
# ----------------------------------------------------------------------------
COMMON_CLASSES: List[str] = [
    "anger", "disgust", "fear",
    "happiness", "neutral", "sadness", "surprise"
]
IMG_SIZE, BATCH_SIZE = (224, 224), 8

# Număr de epoci pentru fiecare fază
EPOCHS_HEAD = 30  # antrenare head
EPOCHS_FINE = 30  # fine-tuning
LR_HEAD, LR_FINE = 1e-3, 1e-5
DROPOUT_HEAD, DROPOUT_FINE = 0.10, 0.20
PATIENCE_ES, PATIENCE_LR, MIN_LR = 5, 3, 1e-6
ES_LR_MIN_DELTA = 0.003
FT_LR_DECAY_STEP = 80.0
FT_LR_DECAY_RATE = 1
FT_DROPOUT = 0.2
FT_ES_PATIENCE = 20  # Adăugat pentru fine-tuning

BASE_DIR  = pathlib.Path(__file__).resolve().parents[2]
DATA_DIR  = BASE_DIR / "data"
TRUNCATED_DIR = DATA_DIR / "truncated"
MODEL_DIR = pathlib.Path("models"); MODEL_DIR.mkdir(exist_ok=True)

# ----------------------------------------------------------------------------
# 1.  GPU / DirectML init
# ----------------------------------------------------------------------------
print("→ Detectăm GPU…", flush=True)
for g in tf.config.list_physical_devices("GPU"):
    tf.config.experimental.set_memory_growth(g, True)
print(f"   {len(tf.config.list_physical_devices('GPU'))} GPU găsit(e).")

strategy = tf.distribute.MirroredStrategy()
print(f"   Numărul replicilor: {strategy.num_replicas_in_sync}")

# ----------------------------------------------------------------------------
# 2.  Datagens
# ----------------------------------------------------------------------------

# Augmentare identică cu originalul
AUG_LAYER = tf.keras.Sequential([
    tf.keras.layers.RandomFlip(mode='horizontal'),
    tf.keras.layers.RandomContrast(0.3)
], name="augmentation")

def load_preprocess(path, augment=False):
    """Încarcă și preprocesează o imagine, cu augmentare opțională"""
    img = tf.io.read_file(path)
    img = tf.image.decode_jpeg(img, channels=3)
    img = tf.image.resize(img, IMG_SIZE)
    img = preprocess_input(img)
    if augment:
        img = AUG_LAYER(img)
    return img

def make_dataset(generator, augment=False):
    """Creează un dataset tf.data dintr-un generator"""
    ds = tf.data.Dataset.from_tensor_slices(generator.filepaths)

    # Construim un StaticHashTable pentru class_indices
    keys = tf.constant(list(generator.class_indices.keys()))
    values = tf.constant(list(generator.class_indices.values()), dtype=tf.int64)
    table = tf.lookup.StaticHashTable(
        tf.lookup.KeyValueTensorInitializer(keys, values), default_value=-1)

    def process_path(path):
        # Extragem numele directorului părinte
        parts = tf.strings.split(path, os.sep)
        parent_name = parts[-2]
        img = load_preprocess(path, augment=augment)
        label = table.lookup(parent_name)
        return img, label

    ds = ds.map(process_path)
    return ds

def build_datagens():
    """Construiește generatoarele de date și dataset-urile echilibrate"""
    aug_train = ImageDataGenerator()
    aug_eval  = ImageDataGenerator()

    # Generatoare pentru validare și test
    val_gen_fer = aug_eval.flow_from_directory(TRUNCATED_DIR/"fer-plus"/"val", target_size=IMG_SIZE, batch_size=BATCH_SIZE,
                                           classes=COMMON_CLASSES, class_mode="sparse", shuffle=False)
    val_gen_raf = aug_eval.flow_from_directory(TRUNCATED_DIR/"rafdb"/"val", target_size=IMG_SIZE, batch_size=BATCH_SIZE,
                                           classes=COMMON_CLASSES, class_mode="sparse", shuffle=False)
    val_gen_aff = aug_eval.flow_from_directory(TRUNCATED_DIR/"affectnet"/"val", target_size=IMG_SIZE, batch_size=BATCH_SIZE,
                                           classes=COMMON_CLASSES, class_mode="sparse", shuffle=False)
    
    # Creăm un set de validare determinist din toate seturile
    val_ds_fer = make_dataset(val_gen_fer)
    val_ds_raf = make_dataset(val_gen_raf)
    val_ds_aff = make_dataset(val_gen_aff)

    # Concatenăm în lanț
    val_ds = val_ds_fer.concatenate(val_ds_raf).concatenate(val_ds_aff)

    # Batching + prefetch la final
    val_ds = val_ds.batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)

    tests = {
        "fer": aug_eval.flow_from_directory(TRUNCATED_DIR/"fer-plus"/"test", target_size=IMG_SIZE, batch_size=BATCH_SIZE,
                                             classes=COMMON_CLASSES, class_mode="sparse", shuffle=False),
        "raf": aug_eval.flow_from_directory(TRUNCATED_DIR/"rafdb"/"test", target_size=IMG_SIZE, batch_size=BATCH_SIZE,
                                             classes=COMMON_CLASSES, class_mode="sparse", shuffle=False),
        "affectnet": aug_eval.flow_from_directory(TRUNCATED_DIR/"affectnet"/"test", target_size=IMG_SIZE, batch_size=BATCH_SIZE,
                                                   classes=COMMON_CLASSES, class_mode="sparse", shuffle=False)
    }

    # Generatoare pentru antrenare
    gen_fer = aug_train.flow_from_directory(TRUNCATED_DIR/"fer-plus"/"train", target_size=IMG_SIZE, batch_size=BATCH_SIZE,
                                           classes=COMMON_CLASSES, class_mode="sparse", shuffle=True)
    gen_raf = aug_train.flow_from_directory(TRUNCATED_DIR/"rafdb"/"train", target_size=IMG_SIZE, batch_size=BATCH_SIZE,
                                           classes=COMMON_CLASSES, class_mode="sparse", shuffle=True)
    gen_aff = aug_train.flow_from_directory(TRUNCATED_DIR/"affectnet"/"train", target_size=IMG_SIZE, batch_size=BATCH_SIZE,
                                           classes=COMMON_CLASSES, class_mode="sparse", shuffle=True)

    # Calculăm numărul de pași per epocă bazat pe numărul real de imagini
    global STEPS_PER_EPOCH
    STEPS_PER_EPOCH = sum(len(g) for g in (gen_fer, gen_raf, gen_aff))
    print(f"\n→ Calculăm STEPS_PER_EPOCH: {STEPS_PER_EPOCH} pași")
    print(f"   ({(STEPS_PER_EPOCH * BATCH_SIZE):,} imagini per epocă)")

    # Creăm dataset-uri tf.data pentru antrenare
    datasets = [make_dataset(g, augment=True) for g in (gen_fer, gen_raf, gen_aff)]
    mixed_ds = tf.data.Dataset.sample_from_datasets(datasets, weights=[1/3, 1/3, 1/3])

    # Creăm dataset-ul final pentru antrenare
    balanced_ds = (
        mixed_ds
        .repeat()                                # nu se mai termină
        .batch(BATCH_SIZE)
        .prefetch(tf.data.AUTOTUNE)
    )

    # Verificăm forma datelor
    print("\n→ Verificăm forma datelor...")
    for img, label in balanced_ds.take(1):
        print(f"   img.shape:  {img.shape}")   # ar trebui (8, 224, 224, 3)
        print(f"   label.shape: {label.shape}") # ar trebui (8,)
        print(f"   label.dtype: {label.dtype}") # ar trebui int32 sau int64

    # Afișăm informații despre setul de validare
    total_fer = tf.data.experimental.cardinality(make_dataset(val_gen_fer)).numpy()
    total_raf = tf.data.experimental.cardinality(make_dataset(val_gen_raf)).numpy()
    total_aff = tf.data.experimental.cardinality(make_dataset(val_gen_aff)).numpy()
    print("\n→ Set de validare:")
    print(f"   FER+/val: {total_fer} imagini")
    print(f"   RAF/val: {total_raf} imagini")
    print(f"   AffectNet/val: {total_aff} imagini")
    print(f"   Total: {total_fer + total_raf + total_aff} imagini")

    return balanced_ds, val_ds, tests, [gen_fer, gen_raf, gen_aff]


def class_weights(gens):
    cnt = np.zeros(len(COMMON_CLASSES), dtype=np.int64)
    for g in gens: cnt += np.bincount(g.classes, minlength=len(COMMON_CLASSES))
    inv = 1.0 / np.where(cnt==0, 1, cnt/cnt.sum()); inv /= inv.max()
    return {i: float(inv[i]) for i in range(len(COMMON_CLASSES))}

# ----------------------------------------------------------------------------
# 4.  Model
# ----------------------------------------------------------------------------

def build_model():
    """Construiește modelul Patt‑Lite cu MobileNet backbone"""
    inputs = keras.Input(shape=(*IMG_SIZE, 3), name="input")

    backbone = keras.applications.MobileNet(
        input_shape=(224, 224, 3), include_top=False, weights="imagenet")
    backbone_out = backbone.layers[-29].output    # 14×14
    backbone = keras.Model(backbone.input, backbone_out, name="mobilenet_backbone")
    backbone.trainable = False

    x = backbone(inputs, training=False)
    
    # Patch extraction layers
    patch_extraction = keras.Sequential([
        keras.layers.SeparableConv2D(256, kernel_size=4, strides=4, padding="same", activation="relu"),
        keras.layers.SeparableConv2D(256, kernel_size=2, strides=2, padding="valid", activation="relu"),
        keras.layers.Conv2D(256, kernel_size=1, strides=1, padding="valid", activation="relu")
    ], name="patch_extraction")
    x = patch_extraction(x)
    
    # Global Average Pooling
    x = keras.layers.GlobalAveragePooling2D()(x)
    
    # Pre-classification block
    x = keras.layers.Dropout(DROPOUT_HEAD)(x)
    x = keras.layers.Dense(32, activation="relu")(x)
    x = keras.layers.BatchNormalization()(x)
    
    # Attention layer
    x = keras.layers.Attention(use_scale=True)([x, x])
    
    # Classification head
    outputs = keras.layers.Dense(len(COMMON_CLASSES), activation="softmax", dtype="float32")(x)

    model = keras.Model(inputs, outputs, name="patt_lite")
    print("Model summary:")
    model.summary()
    return model

def build_finetune_model(base_model, trained_model):
    """Construiește modelul pentru fine-tuning, reutilizând straturile antrenate"""
    inputs = keras.Input(shape=(*IMG_SIZE, 3), name="input")
    
    # Reutilizăm straturile antrenate din modelul anterior
    x = base_model(inputs, training=False)
    
    # Reutilizăm straturile de patch extraction din modelul antrenat
    patch_extraction = trained_model.get_layer("patch_extraction")
    patch_extraction.trainable = False  # Setăm explicit trainable=False
    x = patch_extraction(x)
    
    # Spatial Dropout pentru fine-tuning (pe tensor 4D)
    x = keras.layers.SpatialDropout2D(FT_DROPOUT)(x)
    
    # Global Average Pooling
    x = keras.layers.GlobalAveragePooling2D()(x)
    
    # Pre-classification block cu dropout
    x = keras.layers.Dropout(FT_DROPOUT)(x)
    x = keras.layers.Dense(32, activation="relu")(x)
    x = keras.layers.BatchNormalization()(x)
    
    # Attention layer
    x = keras.layers.Attention(use_scale=True)([x, x])
    
    # Dropout final
    x = keras.layers.Dropout(FT_DROPOUT)(x)
    
    # Classification head
    outputs = keras.layers.Dense(len(COMMON_CLASSES), activation="softmax", dtype="float32")(x)

    model = keras.Model(inputs, outputs, name="patt_lite_finetune")
    return model

# ----------------------------------------------------------------------------
# 5.  Train
# ----------------------------------------------------------------------------

def train():
    train_gen, val_ds, tests, sub_gens = build_datagens()

    # Calculăm ponderile claselor
    weights = class_weights(sub_gens)
    print("\n→ Ponderile claselor:")
    for i, w in weights.items():
        print(f"   {COMMON_CLASSES[i]:10s}: {w:.3f}")

    with strategy.scope():
        model = build_model()
        model.compile(optimizer=keras.optimizers.Adam(learning_rate=LR_HEAD, global_clipnorm=3.0),
                      loss="sparse_categorical_crossentropy",
                      metrics=["sparse_categorical_accuracy"])

    # Configurare TensorBoard
    log_dir = f"logs/{datetime.datetime.now():%Y%m%d-%H%M%S}"
    tensorboard_callback = keras.callbacks.TensorBoard(
        log_dir=log_dir,
        histogram_freq=1,
        write_graph=True,
        write_images=True,
        update_freq='epoch',
        profile_batch=2
    )

    # Faza 1: Head - folosim val_sparse_categorical_accuracy pentru monitorizare
    cb = [
        keras.callbacks.EarlyStopping(monitor="val_sparse_categorical_accuracy", patience=PATIENCE_ES, 
                                    min_delta=ES_LR_MIN_DELTA, restore_best_weights=True),
        keras.callbacks.ReduceLROnPlateau(monitor="val_sparse_categorical_accuracy", factor=0.5,
                                        patience=PATIENCE_LR, min_lr=MIN_LR),
        tensorboard_callback
    ]

    print("\n→ Faza 1: head…")
    history = model.fit(train_gen, epochs=EPOCHS_HEAD, validation_data=val_ds, verbose=1,
                      steps_per_epoch=STEPS_PER_EPOCH, callbacks=cb,
                      class_weight=weights)

    # Fine‑tune ultimele 59 straturi ale backbone‑ului
    print("\n→ Fine‑tuning…")
    unfreeze = 59
    backbone = model.get_layer("mobilenet_backbone")
    backbone.trainable = True
    fine_tune_from = len(backbone.layers) - unfreeze
    for layer in backbone.layers[:fine_tune_from]:
        layer.trainable = False
    for layer in backbone.layers[fine_tune_from:]:
        if isinstance(layer, tf.keras.layers.BatchNormalization):
            layer.trainable = False

    # Construim noul model pentru fine-tuning
    model = build_finetune_model(backbone, model)

    # InverseTimeDecay pentru learning rate în faza fine-tuning
    scheduler = keras.optimizers.schedules.InverseTimeDecay(
        initial_learning_rate=LR_FINE,
        decay_steps=FT_LR_DECAY_STEP,
        decay_rate=FT_LR_DECAY_RATE
    )

    with strategy.scope():
        model.compile(optimizer=keras.optimizers.Adam(learning_rate=scheduler, global_clipnorm=3.0),
                      loss="sparse_categorical_crossentropy",
                      metrics=["sparse_categorical_accuracy"])

    # Configurare TensorBoard pentru fine-tuning
    log_dir_ft = f"logs/{datetime.datetime.now():%Y%m%d-%H%M%S}_ft"
    tensorboard_callback_ft = keras.callbacks.TensorBoard(
        log_dir=log_dir_ft,
        histogram_freq=1,
        write_graph=True,
        write_images=True,
        update_freq='epoch',
        profile_batch=2
    )

    # Faza 2: Fine-tune - folosim val_accuracy pentru a monitoriza generalizarea
    cb = [
        keras.callbacks.EarlyStopping(monitor="val_accuracy", min_delta=ES_LR_MIN_DELTA, 
                                    patience=FT_ES_PATIENCE, restore_best_weights=True),
        tensorboard_callback_ft
    ]

    # Calculăm epoca de start și finală pentru fine-tuning
    start_ft = len(history.epoch)  # numărul efectiv de epoci rulate în faza 1
    end_ft = start_ft + EPOCHS_FINE  # epoca finală pentru fine-tuning

    print("\n→ Faza 2: fine‑tune…")
    model.fit(train_gen, epochs=end_ft, validation_data=val_ds, verbose=1,
              steps_per_epoch=STEPS_PER_EPOCH, initial_epoch=start_ft, callbacks=cb,
              class_weight=weights)

    model.save(MODEL_DIR/"patt_lite_trained.h5")
    print("\n→ Evaluare finală:")
    for name, gen in tests.items():
        _, acc = model.evaluate(gen, verbose=1)
        print(f"   {name.upper():9s}: acc = {acc:.3f}")

if __name__ == "__main__":
    train()
