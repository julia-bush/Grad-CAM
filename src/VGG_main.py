from argparse import ArgumentParser
from pathlib import Path

import tensorflow as tf
from keras_preprocessing.image import ImageDataGenerator
from tensorflow.compat.v1.keras.backend import set_session
from tensorflow.keras import Sequential, optimizers
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.layers import Dense, Dropout, Flatten

from utils import setup_directories, show_classification_report, save_classification_report, predictions_with_truths, \
    save_generator_truths, save_history_results


def run(dataset_name: str, epochs: int, experiment_summary: str = "", finetune_net: str = "") -> None:
    # GPU config works for both one or two GPUs
    physical_devices = tf.config.experimental.list_physical_devices("GPU")
    assert len(physical_devices) > 0, "Not enough GPU hardware devices available"
    config = tf.compat.v1.ConfigProto()
    config.gpu_options.allow_growth = True
    sess = tf.compat.v1.Session(config=config)
    set_session(sess)

    img_size = (224, 224, 3)

    main_dir, train_dir, model_dir, results_dir, pred_dir, n_classes = setup_directories(dataset_name=dataset_name, file_path=Path(__file__).resolve(), experiment_summary=experiment_summary)

    if finetune_net:
        weights=None
        tranianble=True
        if finetune_net == "default":
            finetune_net = f"{experiment_summary[:-9]}/model.hdf5"
    else:
        weights="imagenet"
        tranianble = False

    vgg_conv = tf.keras.applications.VGG16(
        include_top=False,
        weights=weights,
        input_tensor=None,
        input_shape=img_size,
        pooling=None
    )
    print(vgg_conv.summary())

    model = Sequential()

    for layer in vgg_conv.layers[:]:
        layer.trainable = tranianble
        model.add(layer)

    model.add(Flatten())
    model.add(Dropout(0.5))
    model.add(Dense(128, activation="relu"))
    model.add(Dropout(0.5))  # increase?
    model.add(Dense(n_classes, activation="softmax"))

    print(model.summary())

    if finetune_net:
        model.load_weights(model_dir.parent / finetune_net)

    # Batches of tensor image data
    train_datagen = ImageDataGenerator(rescale=1.0 / 255, validation_split=0.2)

    # Change the batchsize according to your system RAM
    train_batchsize = 32
    val_batchsize = 32

    # Data generator for training data
    train_generator = train_datagen.flow_from_directory(
        train_dir,
        target_size=img_size[:-1],
        batch_size=train_batchsize,
        class_mode="categorical",
        shuffle=True,
        seed=0,
        subset="training",
    )

    # Data generator for validation data
    validation_generator = train_datagen.flow_from_directory(
        train_dir,
        target_size=img_size[:-1],
        batch_size=val_batchsize,
        class_mode="categorical",
        shuffle=True,
        seed=0,
        subset="validation",
    )

    save_generator_truths(validation_generator=validation_generator, pred_dir=pred_dir)

    model.compile(
        loss="categorical_crossentropy",
        optimizer=optimizers.Adam(lr=1e-4),
        metrics=["acc"],
    )

    callbacks = [
        ModelCheckpoint(
            model_dir / "model.hdf5", verbose=1, save_weights_only=False, save_best_only=True
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

    save_history_results(train_history=history.history["loss"], val_history=history.history["val_loss"],
                         results_dir=results_dir)

    predictions, truths = predictions_with_truths(model, validation_generator)

    class_names = sorted(set(truths).union(set(predictions)))
    save_classification_report(y_true=truths, y_pred=predictions, results_dir=results_dir, class_names=class_names)
    show_classification_report(y_true=truths, y_pred=predictions, class_names=class_names)


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument("--dataset", default="multiclass_main", type=str)
    parser.add_argument("--epochs", default=10, type=int)
    parser.add_argument("--summary", default="VGG16_trans_double_dropout",
                        help="key to identify experiment when comparing to other runs on the same dataset")
    parser.add_argument("--finetune", default="",
                        help="default: same dataset and experiment_summary; empty to disable finetuning")
    args = parser.parse_args()
    run(dataset_name=args.dataset,
        epochs=args.epochs,
        experiment_summary=args.summary + ("_finetune" if args.finetune else ""),
        finetune_net=args.finetune)
