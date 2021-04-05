import os
from pathlib import Path

import numpy as np
import tensorflow as tf
from tensorflow import keras

from grad_cam_utils import get_img_array, make_gradcam_heatmap, colour_heatmap, superimpose_heatmap, save_superimposed_heatmap

"""
Once you have a trained classifier model, you can run classification predictions and visualise which image regions have
been the most influential for a particular predicted class using Grad-CAM. Place the images to classify in test_dir,
specify: pred_dir, trained_model, class_labels, img_size, last_conv_layer_name, preprocess_input, and hit play.
For reference:
https://keras.io/examples/vision/grad_cam/
https://www.tensorflow.org/guide/autodiff
"""

dataset_name = "HE_defects_sample"
test_dir = Path.cwd().parent / "data" / dataset_name
pred_dir = Path.cwd().parent / "predictions" / dataset_name / "trans_CAM"
Path(pred_dir).mkdir(parents=True, exist_ok=True)
trained_model = f"{Path.cwd().parent}/models/VGG_trans-multiclass_main.hdf5"

model = tf.keras.models.load_model(trained_model)
model.layers[-1].activation = None  # remove softmax to circumvent near-zero probabilities
class_labels = ["corrosion", "crack", "spalling"]  # TODO: read these from file saved during model training
img_size = (224, 224)
preprocess_input = keras.applications.vgg16.preprocess_input
last_conv_layer_name = "block5_conv3"

for filename in os.listdir(test_dir):

    img_path = f"{test_dir}/{filename}"
    img_array = preprocess_input(get_img_array(img_path, size=img_size))

    preds = model.predict(img_array)
    probas = tf.nn.softmax(preds).numpy()
    top_two_preds_indices = np.argsort(-preds)[0][:2]

    for idx, pred_index in enumerate(top_two_preds_indices):
        cam_path = f"{pred_dir}/{idx}_{filename}"  # TODO: decide how to index filenames for different predicted classes
        heatmap = make_gradcam_heatmap(img_array=img_array, model=model, last_conv_layer_name=last_conv_layer_name, pred_index=pred_index)
        legend = f"Predicted {class_labels[pred_index]} with probability {probas[0][pred_index]:.4f}"
        save_superimposed_heatmap(img_path=img_path, heatmap=heatmap, cam_path=cam_path, legend=legend)
