from argparse import ArgumentParser

import tensorflow as tf
from pathlib import Path
from tensorflow.compat.v1.keras.backend import set_session
from tensorflow.keras.callbacks import ModelCheckpoint

from mobilenet import make_mobilenet_with_new_head
from utils import (make_data_generators, plot_learning_curve,
                   predictions_with_truths, save_classification_report,
                   setup_directories, show_classification_report)


def run(dataset_name: str, epochs: int) -> None:
    # GPU config works for both one or two GPUs
    physical_devices = tf.config.experimental.list_physical_devices("GPU")
    assert len(physical_devices) > 0, "Not enough GPU hardware devices available"
    config = tf.compat.v1.ConfigProto()
    config.gpu_options.allow_growth = True
    sess = tf.compat.v1.Session(config=config)
    set_session(sess)

    img_size = (224, 224, 3)

    nn_name = "mobilenet_trans"

    main_dir, train_dir, model_dir, results_dir, pred_dir, n_classes = setup_directories(dataset_name=dataset_name, nn_name=nn_name, file_path=Path(__file__).resolve())

    # Change the batchsize according to your system RAM
    train_batchsize = 4
    val_batchsize = 4

    model = make_mobilenet_with_new_head(n_classes)

    train_generator, validation_generator = make_data_generators(train_dir, img_size[:-1], train_batchsize, val_batchsize, tf.keras.applications.mobilenet.preprocess_input)

    tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=results_dir / "tensorboard", histogram_freq=0, write_graph=True, write_images=False)

    callbacks = [
        ModelCheckpoint(
            model_dir / f"{nn_name}-{dataset_name}.hdf5", verbose=1, save_weights_only=False, save_best_only=True
        ),
        tensorboard_callback
    ]

    history = model.fit_generator(
        generator=train_generator,
        steps_per_epoch=train_generator.samples // train_generator.batch_size,
        # steps_per_epoch=100,
        epochs=epochs,
        verbose=1,
        callbacks=callbacks,
        validation_data=validation_generator,
        validation_steps=validation_generator.samples // validation_generator.batch_size
        # validation_steps=100
    )

    plot_learning_curve(train_history=history.history["loss"]["loss"], val_history=history.history["loss"]["val_loss"], results_dir=results_dir)

    predictions, truths = predictions_with_truths(model, validation_generator)

    class_names = sorted(set(truths).union(set(predictions)))
    save_classification_report(y_true=truths, y_pred=predictions, results_dir=results_dir, class_names=class_names)
    show_classification_report(y_true=truths, y_pred=predictions, class_names=class_names)


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument("--dataset", default="multiclass_main", type=str)
    parser.add_argument("--epochs", default=10, type=int)
    args = parser.parse_args()
    run(dataset_name=args.dataset, epochs=args.epochs)


