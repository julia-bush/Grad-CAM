from pathlib import Path
from typing import List, Tuple

import numpy as np
import pandas as pd
import tensorflow as tf
import tensorflow.keras
import tensorflow.keras.backend as K
from sklearn.metrics import confusion_matrix, classification_report
from tensorflow.keras.applications.vgg16 import (
    VGG16,
)
from tensorflow.python.framework import ops
from tensorflow.python.keras.preprocessing.image import DirectoryIterator
from tqdm import tqdm


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


def setup_directories(dataset_name, nn_name, file_path):
    main_dir = file_path.parent.parent
    train_dir = main_dir / "data" / dataset_name
    n_classes = len(folder_names_in_path(train_dir))
    model_dir = main_dir / "models" / dataset_name / nn_name
    Path(model_dir).mkdir(parents=True, exist_ok=True)
    results_dir = main_dir / "results" / dataset_name / nn_name
    Path(results_dir).mkdir(parents=True, exist_ok=True)
    pred_dir = main_dir / "predictions/" / dataset_name / nn_name
    Path(pred_dir).mkdir(parents=True, exist_ok=True)
    return main_dir, train_dir, model_dir, results_dir, pred_dir, n_classes


def folder_names_in_path(path: Path) -> List[str]:
    return [f.name for f in path.glob("*") if f.is_dir]


def _conf_mat(y_true: List[str], y_pred: List[str], class_names:List[str]) -> np.ndarray:
    return np.array([[sum((np.array(y_true) == i) & (np.array(y_pred) == j)) for i in class_names] for j in class_names])


def show_classification_report(y_true: List[str], y_pred: List[str], class_names: List[str]) -> None:
    matrix = _conf_mat(y_true=y_true, y_pred=y_pred, class_names=class_names)
    print('Confusion Matrix:')
    print(matrix)
    print('Classification Report')
    print(classification_report(y_true=y_true, y_pred=y_pred, zero_division=0))


def save_classification_report(y_true: List[str], y_pred: List[str], results_dir: Path, class_names: List[str]) -> None:
    matrix = _conf_mat(y_true=y_true, y_pred=y_pred, class_names=class_names)
    np.savetxt(f"{results_dir}/confusion_matrix.csv", matrix, delimiter=",")
    pd.DataFrame(matrix, index=class_names, columns=class_names).to_csv(f"{results_dir}/confusion_matrix_headers.csv", index=True)
    report = classification_report(y_true, y_pred, output_dict=True, zero_division=0)
    pd.DataFrame.from_dict(report, orient='columns', dtype=None, columns=None).to_csv(f"{results_dir}/classification_metrics.csv", index=True)


def _get_ordered_class_names(train_generator: DirectoryIterator) -> List[str]:
    labels_dict = train_generator.class_indices
    n_classes = len(labels_dict)
    return [next(k for k, v in labels_dict.items() if v == i) for i in range(n_classes)]


def predictions_with_truths(model: tf.keras.Model, validation_generator: DirectoryIterator) -> Tuple[List[str], List[str]]:
    predictions: List[str] = []
    truths: List[str] = []
    label_number_to_name_map = {number: name for name, number in validation_generator.class_indices.items()}
    for image, label in tqdm(validation_generator):
        predictions += [label_number_to_name_map[label_number] for label_number in np.argmax(model.predict(image), axis=1)]
        truths += [label_number_to_name_map[label_number] for label_number in np.argmax(label, axis=1)]
        if len(predictions) >= len(validation_generator.classes):
            break
    return predictions, truths


def save_generator_truths(validation_generator: DirectoryIterator, pred_dir: Path) -> None:
    """ Records which dataset instances were set aside for validation by validation_split of the ImageDataGenerator.
    After the model is trained and saved, load with load_generator_truths and use to run predictions on (samples taken
    from) validation instances only (and not those instances which had been used for training)."""
    filenames = validation_generator.filenames
    np.save(f"{pred_dir}/val_filenames.npy", filenames)
    classes = validation_generator.classes
    np.save(f"{pred_dir}/val_classes.npy", classes)
    class_indices = {v: k for k, v in validation_generator.class_indices.items()}
    np.save(f"{pred_dir}/val_class_indices.npy", class_indices)
    labels = [class_indices[x] for x in classes]
    np.save(f"{pred_dir}/val_labels.npy", labels)


def load_generator_truths(pred_dir: Path):  # TODO: -> List[str], List[int], Dict[int, str], List[str]
    filenames = np.load(f"{pred_dir}/val_filenames.npy").tolist()
    classes = np.load(f"{pred_dir}/val_classes.npy").tolist()
    class_indices = np.load(f"{pred_dir}/val_class_indices.npy", allow_pickle=True).item()
    labels = np.load(f"{pred_dir}/val_labels.npy").tolist()
    return filenames, classes, class_indices, labels


# def save_generator_predictions(validation_generator: DirectoryIterator, pred_dir: Path) -> None:
#     """ Records predicted classes for dataset instances which were set aside for validation by validation_split of the
#     ImageDataGenerator."""
#     for X_val, y_val in validation_generator:


# def save_img_with_prediction(pred_dir, filenames, idx):
