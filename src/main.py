import os
import sys
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

# TODO: set this up for two GPUs
# cheat no.1
physical_devices = tf.config.experimental.list_physical_devices("GPU")
assert len(physical_devices) > 0, "Not enough GPU hardware devices available"

# # for one GPU
# config = tf.config.experimental.set_memory_growth(physical_devices[0], True)
# sess = tf.compat.v1.Session(config=config)

# # for two GPUs #1
# config = tf.ConfigProto()
# config.gpu_options.allow_growth=True
# sess = tf.Session(config=config)

# for two GPUs #2
config = tf.compat.v1.ConfigProto()
sess = tf.compat.v1.Session

set_session(sess)

# cheat no.2
tf.compat.v1.disable_eager_execution()

# test_dir = sys.argv[1]
dataset_name = "viaducts"
# test_dir = Path.cwd().parent / "data" / dataset_name
# pred_dir = Path.cwd().parent / "predictions" / dataset_name
test_dir = Path.cwd() / "data" / dataset_name
pred_dir = Path.cwd() / "predictions" / dataset_name
Path(pred_dir).mkdir(parents=True, exist_ok=True)

model = VGG16(weights="imagenet")

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
    # cv2.imwrite("gradcam.jpg", cam)
    cv2.imwrite(f"{pred_dir}/{Path(filename).stem}_gradcam.jpg", cam)

    # register_gradient()
    # guided_model = modify_backprop(model, "GuidedBackProp")
    # saliency_fn = compile_saliency_function(guided_model)
    # saliency = saliency_fn([preprocessed_input, 0])
    # gradcam = saliency[0] * heatmap[..., np.newaxis]
    # # cv2.imwrite("guided_gradcam.jpg", deprocess_image(gradcam))
    # cv2.imwrite(
    #     pred_dir + Path(filename).stem + "_guided_gradcam.jpg", deprocess_image(gradcam)
    # )
