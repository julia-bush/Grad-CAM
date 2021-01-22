import os
from pathlib import Path

import cv2
import numpy as np
import tensorflow as tf
from gradcam import grad_cam
from img_process import load_image
from tensorflow.compat.v1.keras.backend import set_session
from tensorflow.keras.applications.vgg16 import (
    VGG16,
    decode_predictions,
)

# GPU config works for both one or two GPUs
physical_devices = tf.config.experimental.list_physical_devices("GPU")
assert len(physical_devices) > 0, "Not enough GPU hardware devices available"
config = tf.compat.v1.ConfigProto()
sess = tf.compat.v1.Session
set_session(sess)

tf.compat.v1.disable_eager_execution()

"""
Specify directory containing images you want to plot grad-CAM heatmaps for.
Uses VGG16 pre-trained on ImageNet, 1000 classes.
"""

dataset_name = "viaducts"
test_dir = Path.cwd().parent / "data" / dataset_name
pred_dir = Path.cwd().parent / "predictions" / dataset_name
Path(pred_dir).mkdir(parents=True, exist_ok=True)

model = VGG16(weights="imagenet")
no_classes = 1000

for filename in os.listdir(test_dir):

    preprocessed_input = load_image(f"{test_dir}/{filename}")
    predictions = model.predict(preprocessed_input)

    print("File: " + filename)
    top_1 = decode_predictions(predictions)[0][0]
    print("Predicted class 1:")
    print("%s (%s) with probability %.2f" % (top_1[1], top_1[0], top_1[2]))
    top_2 = decode_predictions(predictions)[0][1]
    print("Predicted class 2:")
    print("%s (%s) with probability %.2f" % (top_2[1], top_2[0], top_2[2]))
    top_3 = decode_predictions(predictions)[0][2]
    print("Predicted class 3:")
    print("%s (%s) with probability %.2f" % (top_3[1], top_3[0], top_3[2]))

    predicted_class = np.argmax(predictions)
    cam, heatmap = grad_cam(
        model=model,
        image=preprocessed_input,
        category_index=predicted_class,
        layer_name="block5_conv3",
        no_classes=no_classes
    )
    cam = cv2.putText(
        cam,
        f"Class: {top_1[1]}",
        (5, 199),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5,
        (255, 255, 255),
        1,
        cv2.LINE_AA,
    )
    cam = cv2.putText(
        cam,
        f"Probability: {top_1[2]}",
        (5, 219),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5,
        (255, 255, 255),
        1,
        cv2.LINE_AA,
    )

    cv2.imwrite(f"{pred_dir}/{Path(filename).stem}_gradcam.jpg", cam)
