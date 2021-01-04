from tensorflow.keras.applications.vgg16 import (
    VGG16,
    preprocess_input,
    decode_predictions,
)
from tensorflow.keras.models import Model
from tensorflow.keras.preprocessing import image
from tensorflow.python.keras.layers.core import Lambda
from tensorflow.keras.models import Sequential
from tensorflow.python.framework import ops
import tensorflow.keras.backend as K
import tensorflow as tf
import numpy as np
import tensorflow.keras
import sys
import cv2
import os
from pathlib import Path

from tensorflow.compat.v1.keras.backend import set_session

from img_process import load_image, deprocess_image
from utils import register_gradient, modify_backprop, compile_saliency_function
from gradcam import grad_cam


# cheat no.1
physical_devices = tf.config.experimental.list_physical_devices("GPU")
assert len(physical_devices) > 0, "Not enough GPU hardware devices available"
config = tf.config.experimental.set_memory_growth(physical_devices[0], True)
sess = tf.compat.v1.Session(config=config)
set_session(sess)

# cheat no.2
tf.compat.v1.disable_eager_execution()

test_dir = sys.argv[1]
pred_dir = "predictions/" + Path(test_dir).parts[-1] + "/"
Path(pred_dir).mkdir(parents=True, exist_ok=True)

model = VGG16(weights="imagenet")

for filename in os.listdir(test_dir):

    preprocessed_input = load_image(test_dir + filename)
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
    cv2.imwrite(pred_dir + Path(filename).stem + "_gradcam.jpg", cam)

    register_gradient()
    guided_model = modify_backprop(model, "GuidedBackProp")
    saliency_fn = compile_saliency_function(guided_model)
    saliency = saliency_fn([preprocessed_input, 0])
    gradcam = saliency[0] * heatmap[..., np.newaxis]
    # cv2.imwrite("guided_gradcam.jpg", deprocess_image(gradcam))
    cv2.imwrite(
        pred_dir + Path(filename).stem + "_guided_gradcam.jpg", deprocess_image(gradcam)
    )
