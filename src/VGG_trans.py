from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from keras_preprocessing.image import ImageDataGenerator
from tensorflow.compat.v1.keras.backend import set_session
from tensorflow.keras import Sequential, optimizers
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.layers import Dense, Dropout, Flatten


def run():
    # GPU config works for both one or two GPUs
    physical_devices = tf.config.experimental.list_physical_devices("GPU")
    assert len(physical_devices) > 0, "Not enough GPU hardware devices available"
    config = tf.compat.v1.ConfigProto()
    sess = tf.compat.v1.Session
    set_session(sess)

    n_classes = 161
    img_size = (224, 224, 3)
    # print("image size = ", img_size[:-1])

    dataset_name = "HE_defects"
    # train_dir = Path.cwd().parent / "data" / dataset_name
    # model_dir = Path.cwd().parent / "models"
    train_dir = Path.cwd() / "data" / dataset_name
    print(f"train_dir = {train_dir}")
    model_dir = Path.cwd() / "models"
    print(f"model_dir = {model_dir}")
    Path(model_dir).mkdir(parents=True, exist_ok=True)
    # pred_dir = Path.cwd().parent / "predictions/" / dataset_name
    pred_dir = Path.cwd() / "predictions/" / dataset_name
    print(f"pred_dir = {pred_dir}")
    Path(pred_dir).mkdir(parents=True, exist_ok=True)

    vgg_conv = tf.keras.applications.VGG16(
        include_top=False,
        weights="imagenet",
        input_tensor=None,
        input_shape=img_size,
        pooling=None
    )

    print(vgg_conv.summary())

    model = Sequential()

    # for layer in vgg_conv.layers[:-4]:
    #     layer.trainable = False
    #     model.add(layer)
    #
    # for layer in vgg_conv.layers[-4:]:
    #     layer.trainable = True
    #     model.add(layer)

    for layer in vgg_conv.layers[:]:
        layer.trainable = False
        model.add(layer)

    model.add(Flatten())
    model.add(Dense(128, activation="relu"))
    model.add(Dropout(0.5))
    model.add(Dense(n_classes, activation="softmax"))

    model.summary()

    # Batches of tensor image data
    train_datagen = ImageDataGenerator(rescale=1.0 / 255, validation_split=0.2)

    # Change the batchsize according to your system RAM
    train_batchsize = 32
    val_batchsize = 32
    epochs = 50

    # Data generator for training data
    train_generator = train_datagen.flow_from_directory(
        train_dir,
        target_size=img_size[:-1],
        batch_size=train_batchsize,
        class_mode="categorical",
        subset="training",
    )

    # Data generator for validation data
    validation_generator = train_datagen.flow_from_directory(
        train_dir,
        target_size=img_size[:-1],
        batch_size=val_batchsize,
        class_mode="categorical",
        shuffle=False,
        subset="validation",
    )

    model.compile(
        loss="categorical_crossentropy",
        optimizer=optimizers.Adam(lr=1e-4),
        metrics=["acc"],
    )

    callbacks = [
        ModelCheckpoint(
            model_dir / f"VGG16-{dataset_name}.hdf5", verbose=1, save_weights_only=False, save_best_only=True
        )
    ]

    history = model.fit_generator(
        generator=train_generator,
        steps_per_epoch=train_generator.samples // train_generator.batch_size,
        epochs=epochs,
        verbose=1,
        callbacks=callbacks,
        validation_data=validation_generator,
        validation_steps=validation_generator.samples // validation_generator.batch_size
    )

    plt.figure(figsize=(8, 8))
    plt.title("Learning curve")
    plt.plot(history.history["loss"], label="loss")
    plt.plot(history.history["val_loss"], label="val_loss")
    plt.plot(
        np.argmin(history.history["val_loss"]),
        np.min(history.history["val_loss"]),
        marker="x",
        color="r",
        label="best model",
    )
    plt.xlabel("Epochs")
    plt.ylabel("log_loss")
    plt.legend()
    plt.savefig(f"{pred_dir}/learning_curve.png")
    plt.close()

    predictions = model.predict(
        validation_generator,
        steps=validation_generator.samples / validation_generator.batch_size,
        verbose=1,
    )

    # Save a sample of validation results from random batches:
    sample_no = 1000  # sample_no >= number of batches
    num_batches = validation_generator.n // val_batchsize
    batch_sample_idx = np.random.randint(low=0, high=num_batches, size=sample_no)
    for X_val, y_val in validation_generator:
        if validation_generator.batch_index in batch_sample_idx:
            random_sample_idx = np.random.randint(low=0, high=val_batchsize)
            X_val_sample_img = X_val[random_sample_idx, :]
            random_sample_pred_idx = random_sample_idx + validation_generator.batch_index * val_batchsize
            pred_class = np.argmax(predictions[random_sample_pred_idx])
            pred_label = list(validation_generator.class_indices.keys())[pred_class]
            title = "Prediction : {}, confidence : {:.3f}".format(
                pred_label, predictions[random_sample_pred_idx][pred_class]
            )
            plt.figure(figsize=[7, 7])
            plt.axis("off")
            plt.title(title)
            plt.imshow(X_val_sample_img)
            plt.savefig(f"{pred_dir}/{random_sample_pred_idx}.jpg")
            plt.close()
        if validation_generator.batch_index == num_batches-1:
            break


if __name__ == "__main__":
    run()
