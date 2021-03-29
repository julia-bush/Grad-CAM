import numpy as np
import tensorflow as tf


def prepare_image_for_mobilenet(file):
    img = tf.keras.preprocessing.image.load_img(file, target_size=(224, 224))
    img_array = tf.keras.preprocessing.image.img_to_array(img)
    img_array_expanded_dims = np.expand_dims(img_array, axis=0)
    return tf.keras.applications.mobilenet.preprocess_input(img_array_expanded_dims)


def make_mobilenet_with_new_head(n_classes: int):
    base_net = tf.keras.applications.MobileNet(
        include_top=False,
        weights="imagenet",
        # input_tensor=None,
        # input_shape=img_size,
        # pooling=None
    )
    base_net.trainable = False

    model = tf.keras.Sequential([
        base_net,
        tf.keras.layers.Dropout(0.5),
        tf.keras.layers.GlobalAveragePooling2D(),
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.Dense(n_classes, activation='softmax')
    ])

    model.compile(
        loss="categorical_crossentropy",
        optimizer=tf.keras.optimizers.Adam(lr=1e-4),
        metrics=["acc"]
    )
    print(model.summary())

    return model