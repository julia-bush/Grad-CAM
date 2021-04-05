import numpy as np
import tensorflow as tf
from PIL import ImageDraw, ImageFont
from matplotlib import cm as cm
from tensorflow import keras


"""these functions are called from grad_cam.py only"""


def _get_img_array(img_path, size):
    img = keras.preprocessing.image.load_img(img_path, target_size=size)
    array = keras.preprocessing.image.img_to_array(img)
    array = np.expand_dims(array, axis=0)
    return array


def _make_gradcam_heatmap(img_array, model, last_conv_layer_name, pred_index=None):
    # collect the necessary tf.Variables into keras.Model, subclass of tf.Module, for convenience
    grad_model = tf.keras.models.Model(
        [model.inputs], [model.get_layer(last_conv_layer_name).output, model.output]
    )

    # record forward pass operations on a tf.GradientTape for automatic differentiation
    with tf.GradientTape() as tape:
        last_conv_layer_output, preds = grad_model(img_array)
        if pred_index is None:
            pred_index = tf.argmax(preds[0])
        class_channel = preds[:, pred_index]  # chosen class prediction before softmax TODO: better var name

    # d(class_channel)/d(last_conv_layer_output), i.e.
    # how sensitive is class_channel to changes in last_conv_layer_output
    grads = tape.gradient(class_channel, last_conv_layer_output)
    # take the mean of gradients over every response map
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))

    last_conv_layer_output = last_conv_layer_output[0]
    # heatmap is the weighted (by mean gradient) sum of response maps in chosen layer
    heatmap = last_conv_layer_output @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)
    # normalise between 0 & 1
    heatmap = tf.maximum(heatmap, 0) / tf.math.reduce_max(heatmap)

    return heatmap.numpy()


def _colour_heatmap(heatmap, colourmap="jet"):
    scaled_heatmap = np.uint8(255 * heatmap)
    cmap = cm.get_cmap(colourmap)
    cmap_colors = cmap(np.arange(256))[:, :3]
    cmap_heatmap = cmap_colors[scaled_heatmap]
    return cmap_heatmap


def _superimpose_heatmap(img_path, heatmap, alpha):
    img = keras.preprocessing.image.load_img(img_path)
    img = keras.preprocessing.image.img_to_array(img)
    heatmap = keras.preprocessing.image.array_to_img(heatmap)
    heatmap = heatmap.resize((img.shape[1], img.shape[0]))
    heatmap = keras.preprocessing.image.img_to_array(heatmap)
    superimposed_img = heatmap * alpha + img
    superimposed_img = keras.preprocessing.image.array_to_img(superimposed_img)
    return superimposed_img


def _save_superimposed_heatmap(img_path, heatmap, cam_path, colourmap="jet", alpha=0.4, legend=None):
    cmap_heatmap = _colour_heatmap(heatmap=heatmap, colourmap=colourmap)
    superimposed_img = _superimpose_heatmap(img_path=img_path, heatmap=cmap_heatmap, alpha=alpha)
    if legend is not None:
        draw = ImageDraw.Draw(superimposed_img)
        font = ImageFont.truetype("arial.ttf", 18)
        draw.text((5, 5), legend, (255, 255, 255), font=font)
    superimposed_img.save(cam_path)
