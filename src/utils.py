import numpy as np
import tensorflow as tf
import tensorflow.keras
import tensorflow.keras.backend as K
from tensorflow.keras.applications.vgg16 import (
    VGG16,
)
from tensorflow.python.framework import ops


def setup_tensorflow():
    """ Check GPU is on and configure it to avoid weird errors, using 1 or 2 GPUs """
    physical_devices = tf.config.experimental.list_physical_devices("GPU")
    assert len(physical_devices) > 0, "Not enough GPU hardware devices available"
    config = tf.compat.v1.ConfigProto()
    config.gpu_options.allow_growth = True
    sess = tf.compat.v1.Session(config=config)
    set_session(sess)

    tf.compat.v1.disable_eager_execution()


def normalise(x):
    # utility function to normalise a tensor by its L2 norm
    return x / (K.sqrt(K.mean(K.square(x))) + 1e-5)


def register_gradient():
    if "GuidedBackProp" not in ops._gradient_registry._registry:

        @ops.RegisterGradient("GuidedBackProp")
        def _GuidedBackProp(op, grad):
            dtype = op.inputs[0].dtype
            return (
                grad * tf.cast(grad > 0.0, dtype) * tf.cast(op.inputs[0] > 0.0, dtype)
            )


def compile_saliency_function(model, activation_layer="block5_conv3"):
    input_img = model.input
    layer_dict = dict([(layer.name, layer) for layer in model.layers[1:]])
    layer_output = layer_dict[activation_layer].output
    max_output = K.max(layer_output, axis=3)
    saliency = K.gradients(K.sum(max_output), input_img)[0]
    return K.function([input_img, K.learning_phase()], [saliency])


def modify_backprop(model, name):
    g = tf.compat.v1.get_default_graph()
    with g.gradient_override_map({"Relu": name}):

        # get layers that have an activation
        layer_dict = [
            layer for layer in model.layers[1:] if hasattr(layer, "activation")
        ]

        # replace relu activation
        for layer in layer_dict:
            if layer.activation == tensorflow.keras.activations.relu:
                layer.activation = tf.nn.relu

        # re-instanciate a new model
        new_model = VGG16(weights="imagenet")
    return new_model


def deprocess_image(x):
    """
    Same normalization as in:
    https://github.com/fchollet/keras/blob/master/examples/conv_filter_visualization.py
    """
    if np.ndim(x) > 3:
        x = np.squeeze(x)
    # normalize tensor: center on 0., ensure std is 0.1
    x -= x.mean()
    x /= x.std() + 1e-5
    x *= 0.1

    # clip to [0, 1]
    x += 0.5
    x = np.clip(x, 0, 1)

    # convert to RGB array
    x *= 255
    if K.image_data_format() == "th":
        x = x.transpose((1, 2, 0))
    x = np.clip(x, 0, 255).astype("uint8")
    return x
