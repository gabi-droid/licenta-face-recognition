# ============================================================================
#  Patt‑Lite – train_patt_lite.py  ✧ versiune optimizată (fix fine‑tune layer)
# ----------------------------------------------------------------------------
#  • Corectează numele backbone‑ului MobileNet → "mobilenet_backbone".
#  • Fine‑tuning-ul se face acum pe ultimele 40 de straturi ale acestui backbone.
# ============================================================================

from __future__ import annotations
import os, shutil, datetime, random, pathlib
from typing import Tuple, List, Dict

import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications.mobilenet import preprocess_input

# ----------------------------------------------------------------------------
# CONFIG
# ----------------------------------------------------------------------------
COMMON_CLASSES: List[str] = [
    "anger", "contempt", "disgust", "fear",
    "happiness", "neutral", "sadness", "surprise",
]
IMG_SIZE, BATCH_SIZE = (224, 224), 8

EPOCHS_HEAD, EPOCHS_FINE = 30, 30
LR_HEAD,   LR_FINE      = 1e-3, 1e-5
DROPOUT_HEAD, DROPOUT_FINE = 0.10, 0.20
PATIENCE_ES, PATIENCE_LR, MIN_LR = 5, 3, 1e-6
ES_LR_MIN_DELTA = 0.003

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

# Enable mixed precision
policy = tf.keras.mixed_precision.Policy('mixed_float16')
tf.keras.mixed_precision.set_global_policy(policy)
print(f"   Mixed precision enabled: {policy.name}")

strategy = tf.distribute.MirroredStrategy()
print(f"   Numărul replicilor: {strategy.num_replicas_in_sync}")

# ----------------------------------------------------------------------------
# 2.  Datagens
# ----------------------------------------------------------------------------

def build_datagens():
    aug_train = ImageDataGenerator(preprocessing_function=lambda x: tf.image.resize(preprocess_input(x), IMG_SIZE),
                                   horizontal_flip=True, rotation_range=10,
                                   width_shift_range=0.1, height_shift_range=0.1)
    aug_eval  = ImageDataGenerator(preprocessing_function=lambda x: tf.image.resize(preprocess_input(x), IMG_SIZE))

    val_gen = aug_eval.flow_from_directory(TRUNCATED_DIR/"fer-plus"/"val", target_size=IMG_SIZE, batch_size=BATCH_SIZE,
                                           classes=COMMON_CLASSES, class_mode="categorical", shuffle=False)
    tests = {
        "fer": aug_eval.flow_from_directory(TRUNCATED_DIR/"fer-plus"/"test", target_size=IMG_SIZE, batch_size=BATCH_SIZE,
                                             classes=COMMON_CLASSES, class_mode="categorical", shuffle=False),
        "raf": aug_eval.flow_from_directory(TRUNCATED_DIR/"rafdb"/"test", target_size=IMG_SIZE, batch_size=BATCH_SIZE,
                                             classes=COMMON_CLASSES, class_mode="categorical", shuffle=False),
        "affectnet": aug_eval.flow_from_directory(TRUNCATED_DIR/"affectnet"/"test", target_size=IMG_SIZE, batch_size=BATCH_SIZE,
                                                   classes=COMMON_CLASSES, class_mode="categorical", shuffle=False)
    }

    gen_fer = aug_train.flow_from_directory(TRUNCATED_DIR/"fer-plus"/"train", target_size=IMG_SIZE, batch_size=BATCH_SIZE,
                                           classes=COMMON_CLASSES, class_mode="categorical", shuffle=True)
    gen_raf = aug_train.flow_from_directory(TRUNCATED_DIR/"rafdb"/"train", target_size=IMG_SIZE, batch_size=BATCH_SIZE,
                                           classes=COMMON_CLASSES, class_mode="categorical", shuffle=True)
    gen_aff = aug_train.flow_from_directory(TRUNCATED_DIR/"affectnet"/"train", target_size=IMG_SIZE, batch_size=BATCH_SIZE,
                                           classes=COMMON_CLASSES, class_mode="categorical", shuffle=True)

    class Combined(tf.keras.utils.Sequence):
        def __init__(self, gens):
            self.gens, self.len_max = gens, max(len(g) for g in gens)
        def __len__(self): return self.len_max * len(self.gens)
        def __getitem__(self, idx):
            g = self.gens[idx % len(self.gens)]
            return g[(idx//len(self.gens)) % len(g)]
        def on_epoch_end(self):
            for g in self.gens: g.on_epoch_end()

    return Combined([gen_fer, gen_raf, gen_aff]), val_gen, tests, [gen_fer, gen_raf, gen_aff]


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
    x = keras.layers.experimental.preprocessing.Resizing(224, 224, name="resize")(inputs)

    backbone = keras.applications.MobileNet(
        input_shape=(224, 224, 3), include_top=False, weights="imagenet")
    backbone._name = "mobilenet_backbone"
    backbone.trainable = False

    x = backbone(x, training=False)
    x = keras.layers.SeparableConv2D(256, kernel_size=4, strides=4, padding="same", activation="relu")(x)
    x = keras.layers.SeparableConv2D(256, kernel_size=2, strides=2, padding="valid", activation="relu")(x)
    x = keras.layers.Conv2D(256, kernel_size=1, strides=1, padding="valid", activation="relu")(x)
    x = keras.layers.GlobalAveragePooling2D()(x)
    x = keras.layers.Dropout(DROPOUT_HEAD)(x)
    x = keras.layers.Dense(128, activation="relu")(x); x = keras.layers.BatchNormalization()(x)
    x = keras.layers.Dense(64, activation="relu")(x);  x = keras.layers.BatchNormalization()(x)
    x = keras.layers.Attention(use_scale=True)([x, x])
    outputs = keras.layers.Dense(len(COMMON_CLASSES), activation="softmax", dtype="float32")(x)

    model = keras.Model(inputs, outputs, name="patt_lite")
    print("Model summary:")
    model.summary()
    return model

# ----------------------------------------------------------------------------
# 5.  Train
# ----------------------------------------------------------------------------

def train():
    train_gen, val_gen, tests, sub_gens = build_datagens()
    cw = class_weights(sub_gens); print("\n→ Class‑weights:", cw)

    with strategy.scope():
        model = build_model()
        model.compile(optimizer=keras.optimizers.Adam(learning_rate=LR_HEAD, global_clipnorm=3.0),
                      loss="categorical_crossentropy", metrics=["accuracy"])

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

    cb = [
        keras.callbacks.EarlyStopping(monitor="val_accuracy", patience=PATIENCE_ES, 
                                    min_delta=ES_LR_MIN_DELTA, restore_best_weights=True),
        keras.callbacks.ReduceLROnPlateau(monitor="val_accuracy", patience=PATIENCE_LR,
                                        min_delta=ES_LR_MIN_DELTA, min_lr=MIN_LR, verbose=1),
        tensorboard_callback
    ]

    print("\n→ Faza 1: head…")
    history = model.fit(train_gen, epochs=EPOCHS_HEAD, validation_data=val_gen, verbose=1,
                      class_weight=cw, callbacks=cb)

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

    with strategy.scope():
        model.compile(optimizer=keras.optimizers.Adam(learning_rate=LR_FINE, global_clipnorm=3.0),
                      loss="categorical_crossentropy", metrics=["accuracy"])

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

    cb = [
        keras.callbacks.EarlyStopping(monitor="accuracy", min_delta=ES_LR_MIN_DELTA, 
                                    patience=PATIENCE_ES, restore_best_weights=True),
        tensorboard_callback_ft
    ]

    print("\n→ Faza 2: fine‑tune…")
    model.fit(train_gen, epochs=EPOCHS_FINE, validation_data=val_gen, verbose=1,
              initial_epoch=history.epoch[-PATIENCE_ES], class_weight=cw, callbacks=cb)

    model.save(MODEL_DIR/"patt_lite_trained.h5")
    print("\n→ Evaluare finală:")
    for name, gen in tests.items():
        _, acc = model.evaluate(gen, verbose=1)
        print(f"   {name.upper():9s}: acc = {acc:.3f}")

if __name__ == "__main__":
    train()
