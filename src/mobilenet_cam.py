import os
from pathlib import Path
from typing import List
import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense, Dropout, Flatten
from tensorflow.keras import Sequential, optimizers
from keras_preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
from utils import setup_tensorflow

""" End to end experiment using mobilenet, aiming to extract gradcam heatmaps """

def run():
    setup_tensorflow()
    train(train_dir=Path(__file__).parent.parent / "data/Corrosion", img_size=224)


def make_model(img_size: int, n_classes: int, n_unfrozen_layers: int = 1):
    net = tf.keras.applications.MobileNet(
        include_top=False,
        weights="imagenet",
        input_tensor=None,
        input_shape=(img_size, img_size, 3),
        pooling=None
    )

    data_augmentation = tf.keras.Sequential(
        [
            tf.keras.layers.experimental.preprocessing.RandomFlip("horizontal"),
            tf.keras.layers.experimental.preprocessing.RandomRotation(0.1),
        ]
    )

    i = tf.keras.layers.Input([img_size, img_size, 3], dtype = tf.uint8)
    x = data_augmentation(i)

    x = tf.cast(x, tf.float32)
    x = tf.keras.applications.mobilenet.preprocess_input(x)

    for layer_number, layer in enumerate(net.layers):
        if layer_number <= len(net.layers) - n_unfrozen_layers:
            layer.trainable = False
        x = layer(x)

    x = Flatten()(x)
    x = Dense(128, activation="relu")(x)
    x = Dropout(0.5)(x)
    x = Dense(n_classes, activation="softmax")(x)
    
    model = tf.keras.Model(inputs=[i], outputs=[x])

    model.compile(
        loss="categorical_crossentropy",
        optimizer=optimizers.Adam(lr=1e-4),
        metrics=["acc"],
    )

    return model

def get_subfolder_names(path: Path) -> List[str]:
    return [f.name for f in path.glob("*") if f.is_dir()]


def train(train_dir: Path, img_size: int, train_batchsize: int = 32, validation_split=0.2, epochs: int = 20):

    train_datagen = ImageDataGenerator(validation_split=validation_split)

    # Data generator for training data
    train_generator = train_datagen.flow_from_directory(train_dir, target_size=(img_size, img_size), batch_size=train_batchsize, class_mode="categorical", subset="training")
    validation_generator = train_datagen.flow_from_directory(train_dir, target_size=(img_size, img_size), batch_size=train_batchsize, class_mode="categorical", subset="validation")

    classes = get_subfolder_names(train_dir)
    print(f"Training model for classes: {classes}")
    model = make_model(img_size=img_size, n_classes=len(classes))
    
    checkpoint = ModelCheckpoint(f"models/mobilenet_{train_dir.name}.hdf5", monitor='val_acc', verbose=1, save_best_only=True, save_weights_only=False, mode='auto', period=1)
    early = EarlyStopping(monitor='val_acc', min_delta=0, patience=20, verbose=1, mode='auto')

    history = model.fit(train_generator, validation_data=validation_generator, epochs=epochs, verbose=1, callbacks=[checkpoint, early])
    return model


if __name__ == "__main__": 
    run()
    # TODO: gradcam part

