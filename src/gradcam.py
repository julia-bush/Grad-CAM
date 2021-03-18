import cv2
import numpy as np
import tensorflow as tf
import tensorflow.keras.backend as K
from tensorflow.keras.models import Model
from tensorflow.python.keras.layers.core import Lambda
from utils import normalise


def target_category_loss(x, category_index, no_classes):
    return tf.multiply(x, K.one_hot([category_index], no_classes))


def target_category_loss_output_shape(input_shape):
    return input_shape


def _compute_gradients(tensor, var_list):
    grads = tf.gradients(tensor, var_list)
    return [
        grad if grad is not None else tf.zeros_like(var)
        for var, grad in zip(var_list, grads)
    ]


def grad_cam(model, image, category_index, layer_name, no_classes):

    target_layer = lambda x: target_category_loss(x, category_index, no_classes)
    x = Lambda(target_layer, output_shape=target_category_loss_output_shape)(
        model.output
    )
    model = Model(inputs=model.input, outputs=x)
    # model.summary()
    loss = K.sum(model.output)
    conv_output = model.get_layer(layer_name).output
    grads = normalise(_compute_gradients(loss, [conv_output])[0])
    gradient_function = K.function([model.input], [conv_output, grads])

    output, grads_val = gradient_function([image])
    output, grads_val = output[0, :], grads_val[0, :, :, :]

    weights = np.mean(grads_val, axis=(0, 1))
    cam = np.ones(output.shape[0:2], dtype=np.float32)

    for i, w in enumerate(weights):
        cam += w * output[:, :, i]

    cam = cv2.resize(cam, (224, 224))
    cam = np.maximum(cam, 0)
    heatmap = cam / np.max(cam)

    # Return to BGR [0..255] from the preprocessed image
    image = image[0, :]
    image -= np.min(image)
    image = np.minimum(image, 255)

    cam = cv2.applyColorMap(np.uint8(255 * heatmap), cv2.COLORMAP_JET)
    cam = np.float32(cam) + np.float32(image)
    cam = 255 * cam / np.max(cam)
    return np.uint8(cam), heatmap
