from pathlib import Path

import tensorflow as tf
from keras_preprocessing.image import ImageDataGenerator
from tensorflow.compat.v1.keras.backend import set_session
from tensorflow.keras import Sequential, optimizers
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
from tensorflow.keras.layers import Dense, Flatten, Conv2D, MaxPool2D


def run():
    # GPU config works for both one or two GPUs
    physical_devices = tf.config.experimental.list_physical_devices("GPU")
    assert len(physical_devices) > 0, "Not enough GPU hardware devices available"
    config = tf.compat.v1.ConfigProto()
    sess = tf.compat.v1.Session
    set_session(sess)

    # TODO: feed n_classes into model layer parameters
    n_classes = 2
    img_size = (224, 224, 3)
    train_batchsize = 32
    val_batchsize = 32
    epochs = 100

    dataset_name = "concrete"
    train_dir = Path.cwd().parent / "data" / dataset_name
    model_dir = Path.cwd().parent / "models"
    Path(model_dir).mkdir(parents=True, exist_ok=True)
    pred_dir = Path.cwd().parent / "predictions/" / dataset_name
    Path(pred_dir).mkdir(parents=True, exist_ok=True)

    # Batches of tensor image data
    train_datagen = ImageDataGenerator(rescale=1.0 / 255, validation_split=0.2)

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

    model = Sequential()
    model.add(
        Conv2D(
            input_shape=img_size,
            filters=64,
            kernel_size=(3, 3),
            padding="same",
            activation="relu",
        )
    )
    model.add(Conv2D(filters=64, kernel_size=(3, 3), padding="same", activation="relu"))
    model.add(MaxPool2D(pool_size=(2, 2), strides=(2, 2)))
    model.add(Conv2D(filters=128, kernel_size=(3, 3), padding="same", activation="relu"))
    model.add(Conv2D(filters=128, kernel_size=(3, 3), padding="same", activation="relu"))
    model.add(MaxPool2D(pool_size=(2, 2), strides=(2, 2)))
    model.add(Conv2D(filters=256, kernel_size=(3, 3), padding="same", activation="relu"))
    model.add(Conv2D(filters=256, kernel_size=(3, 3), padding="same", activation="relu"))
    model.add(Conv2D(filters=256, kernel_size=(3, 3), padding="same", activation="relu"))
    model.add(MaxPool2D(pool_size=(2, 2), strides=(2, 2)))
    model.add(Conv2D(filters=512, kernel_size=(3, 3), padding="same", activation="relu"))
    model.add(Conv2D(filters=512, kernel_size=(3, 3), padding="same", activation="relu"))
    model.add(Conv2D(filters=512, kernel_size=(3, 3), padding="same", activation="relu"))
    model.add(MaxPool2D(pool_size=(2, 2), strides=(2, 2)))
    model.add(Conv2D(filters=512, kernel_size=(3, 3), padding="same", activation="relu"))
    model.add(Conv2D(filters=512, kernel_size=(3, 3), padding="same", activation="relu"))
    model.add(Conv2D(filters=512, kernel_size=(3, 3), padding="same", activation="relu"))
    model.add(MaxPool2D(pool_size=(2, 2), strides=(2, 2)))

    model.add(Flatten())
    model.add(Dense(units=4096, activation="relu"))
    model.add(Dense(units=4096, activation="relu"))
    model.add(Dense(units=2, activation="softmax"))

    model.compile(
        loss="categorical_crossentropy",
        optimizer=optimizers.Adam(lr=0.001),  #
        metrics=["accuracy"],
    )

    model.summary()

    checkpoint = ModelCheckpoint(
        f"VGG16-{dataset_name}.hdf5",
        monitor="val_acc",
        verbose=1,
        save_best_only=True,
        save_weights_only=False,
        mode="auto",
        period=1,
    )

    early = EarlyStopping(
        monitor="val_acc", min_delta=0, patience=20, verbose=1, mode="auto"
    )

    history = model.fit_generator(
        generator=train_generator,
        steps_per_epoch=train_generator.samples // train_generator.batch_size,
        epochs=epochs,
        verbose=1,
        callbacks=[checkpoint, early],
        validation_data=validation_generator,
        validation_steps=validation_generator.samples
        // validation_generator.batch_size,
    )

    # Save a sample of validation results from random batches:
    sample_no = 10  # sample_no >= number of batches
    num_batches = validation_generator.n // val_batchsize
    batch_sample_idx = np.random.randint(low=0, high=num_batches, size=sample_no)
    print(f"batch_sample_idx = {batch_sample_idx}")
    for X_val, y_val in validation_generator:
        print(f"validation_generator.batch_index = {validation_generator.batch_index}")
        if validation_generator.batch_index in batch_sample_idx:
            random_sample_idx = np.random.randint(low=0, high=val_batchsize)
            print(f"random_sample_idx = {random_sample_idx}")
            X_val_sample_img = X_val[random_sample_idx, :]
            random_sample_pred_idx = (
                random_sample_idx + validation_generator.batch_index * val_batchsize
            )
            print(f"random_sample_pred_idx = {random_sample_pred_idx}")
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
        if validation_generator.batch_index == num_batches - 1:
            break


if __name__ == "__main__":
    run()
