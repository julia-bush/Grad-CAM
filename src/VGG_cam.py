import os
from pathlib import Path

import cv2
import numpy as np
import tensorflow as tf
from gradcam import grad_cam
from img_process import load_image
from tensorflow.compat.v1.keras.backend import set_session
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense, Dropout, Flatten

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

dataset_name = "concrete_sample"
test_dir = Path.cwd().parent / "data" / dataset_name
pred_dir = Path.cwd().parent / "predictions" / dataset_name / "fine_CAM"
Path(pred_dir).mkdir(parents=True, exist_ok=True)
model_weights = f"{Path.cwd().parent}/models/VGG16-concrete-fine.hdf5"

no_classes = 2
img_size = (224, 224, 3)

vgg_conv = tf.keras.applications.VGG16(
    include_top=False,
    weights=None,
    input_tensor=None,
    input_shape=img_size,
    pooling=None
)

print(vgg_conv.summary())

model = Sequential()

for layer in vgg_conv.layers[:]:
    model.add(layer)

model.add(Flatten())
model.add(Dense(128, activation="relu"))
model.add(Dropout(0.5))
model.add(Dense(no_classes, activation="softmax"))

model.summary()

model.load_weights(model_weights)

for filename in os.listdir(test_dir):

    preprocessed_input = load_image(f"{test_dir}/{filename}")
    predictions = model.predict(preprocessed_input)

    print("File: " + filename)
    print(f"Predicted class: {np.argmax(predictions)} with probability {np.max(predictions)}")

    predicted_class = np.argmax(predictions)
    cam, heatmap = grad_cam(
        model=model,
        image=preprocessed_input,
        category_index=predicted_class,
        layer_name="block5_conv2",
        no_classes=no_classes
    )
    cam = cv2.putText(
        cam,
        f"Class: {np.argmax(predictions)}",
        (5, 199),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5,
        (255, 255, 255),
        1,
        cv2.LINE_AA,
    )
    cam = cv2.putText(
        cam,
        f"Probability: {np.max(predictions)}",
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
