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
    # cheat no.1
    physical_devices = tf.config.experimental.list_physical_devices("GPU")
    assert len(physical_devices) > 0, "Not enough GPU hardware devices available"
    config = tf.config.experimental.set_memory_growth(physical_devices[0], True)
    sess = tf.compat.v1.Session(config=config)
    set_session(sess)

    n_classes = 2
    img_size = (227, 227, 3)
    # print("image size = ", img_size[:-1])

    dataset_name = "concrete"
    train_dir = Path.cwd().parent / "data" / dataset_name
    model_dir = Path.cwd().parent / "models"
    Path(model_dir).mkdir(parents=True, exist_ok=True)
    pred_dir = Path.cwd().parent / "predictions/" + dataset_name + "/"
    Path(pred_dir).mkdir(parents=True, exist_ok=True)

    vgg_conv = tf.keras.applications.VGG16(
        include_top=False,
        weights="imagenet",
        input_tensor=None,
        input_shape=img_size,
        pooling=None,
        classes=1000,
        classifier_activation="softmax",
    )

    for layer in vgg_conv.layers[:]:
        layer.trainable = False

    print(vgg_conv.summary())

    model = Sequential()
    model.add(vgg_conv)
    model.add(Flatten())
    model.add(Dense(128, activation="relu"))
    model.add(Dropout(0.5))
    model.add(Dense(n_classes, activation="softmax"))

    model.summary()

    # Batches of tensor image data
    train_datagen = ImageDataGenerator(rescale=1.0 / 255, validation_split=0.2)

    # Change the batchsize according to your system RAM
    train_batchsize = 2
    val_batchsize = 1
    epochs = 3

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
        optimizer=optimizers.RMSprop(lr=1e-4),
        metrics=["acc"],
    )

    callbacks = [
        ModelCheckpoint(
            model_dir / f"VGG16-{dataset_name}.hdf5", verbose=1, save_weights_only=True
        )
    ]

    history = model.fit(
        train_generator,
        steps_per_epoch=train_generator.samples // train_generator.batch_size,
        epochs=epochs,
        callbacks=callbacks,
        validation_data=validation_generator,
        validation_steps=800,
        verbose=1,
    )

    predictions = model.predict(
        validation_generator,
        steps=validation_generator.samples / validation_generator.batch_size,
        verbose=1,
    )

    # Show validation results
    for i in range(validation_generator.n):
        pred_class = np.argmax(predictions[i])
        pred_label = list(validation_generator.class_indices.keys())[pred_class]

        title = "Prediction : {}, confidence : {:.3f}".format(
            pred_label, predictions[i][pred_class]
        )

        X_val, y_val = next(validation_generator)

        # original = load_img('{}/{}'.format(validation_dir, fnames[errors[i]]))
        plt.figure(figsize=[7, 7])
        plt.axis("off")
        plt.title(title)
        # plt.imshow(original)
        plt.imshow(np.squeeze(X_val, axis=0))
        plt.show()


if __name__ == "__main__":
    run()
