import os
from pathlib import Path
from argparse import ArgumentParser

import cv2
import numpy as np
import tensorflow as tf
from gradcam import grad_cam
from img_process import load_image
from tensorflow.compat.v1.keras.backend import set_session
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense, Dropout, Flatten

from utils import setup_directories, load_generator_truths


def run(dataset_name: str, nn_name: str) -> None:

    # GPU config works for both one or two GPUs
    physical_devices = tf.config.experimental.list_physical_devices("GPU")
    assert len(physical_devices) > 0, "Not enough GPU hardware devices available"
    config = tf.compat.v1.ConfigProto()
    config.gpu_options.allow_growth = True
    sess = tf.compat.v1.Session(config=config)
    set_session(sess)

    tf.compat.v1.disable_eager_execution()

    """
    Specify directory containing images you want to plot grad-CAM heatmaps for.
    Uses VGG16 transfer-learned on the concrete dataset.
    """

    img_size = (224, 224, 3)

    main_dir, train_dir, model_dir, results_dir, pred_dir, n_classes = setup_directories(dataset_name=dataset_name, nn_name=nn_name, file_path=Path(__file__))

    # get the trained model to fine-tune
    trained_model = f"{model_dir}/{nn_name}-{dataset_name}.hdf5"

    model = tf.keras.models.load_model(trained_model)

    model.summary()

    filenames, classes, class_indices, labels = load_generator_truths(pred_dir=pred_dir)

    for idx, filename in enumerate(filenames):
        preprocessed_input = load_image(f"{train_dir}/{filename}")
        predictions = model.predict(preprocessed_input)

        print("File: " + filename)
        print(f"Predicted class: {np.argmax(predictions)} with probability {np.max(predictions)}")

        for prediction_number in range(1,3):  # make predictions for two highest probability classes
            predicted_class_index = np.argsort(np.max(predictions, axis=0))[-prediction_number]
            predicted_class_label = class_indices[predicted_class_index]

            cam, heatmap = grad_cam(
                model=model,
                image=preprocessed_input,
                category_index=predicted_class_index,
                layer_name="block5_conv3",
                no_classes=n_classes
            )
            cam = cv2.putText(
                cam,
                f"Ground truth: {labels[idx]}",
                (5, 179),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (255, 255, 255),
                1,
                cv2.LINE_AA,
            )
            cam = cv2.putText(
                cam,
                f"Prediction {prediction_number}: {predicted_class_label}",
                (5, 199),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (255, 255, 255),
                1,
                cv2.LINE_AA,
            )
            cam = cv2.putText(
                cam,
                f"Probability: {predictions[0][predicted_class_index]:.3f}",
                (5, 219),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (255, 255, 255),
                1,
                cv2.LINE_AA,
            )
            cv2.imwrite(f"{pred_dir}/{Path(filename).stem}_gradcam_{prediction_number}.jpg", cam)


            # register_gradient()
            # guided_model = modify_backprop(model, "GuidedBackProp")
            # saliency_fn = compile_saliency_function(guided_model)
            # saliency = saliency_fn([preprocessed_input, 0])
            # gradcam = saliency[0] * heatmap[..., np.newaxis]
            # # cv2.imwrite("guided_gradcam.jpg", deprocess_image(gradcam))
            # cv2.imwrite(
            #     pred_dir + Path(filename).stem + "_guided_gradcam.jpg", deprocess_image(gradcam)
            # )


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument("--dataset", default="multiclass_main", type=str)
    parser.add_argument("--nn", default="VGG_fine", type=str)
    args = parser.parse_args()
    run(dataset_name=args.dataset, nn_name=args.nn)
