from pathlib import Path
from typing import List, Tuple

import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.metrics import classification_report
from tensorflow.python.keras.preprocessing.image import DirectoryIterator
from tqdm import tqdm


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


def _conf_mat(
    y_true: List[str], y_pred: List[str], class_names: List[str]
) -> np.ndarray:
    return np.array(
        [
            [
                sum((np.array(y_true) == i) & (np.array(y_pred) == j))
                for i in class_names
            ]
            for j in class_names
        ]
    )


def show_classification_report(
    y_true: List[str], y_pred: List[str], class_names: List[str]
) -> None:
    matrix = _conf_mat(y_true=y_true, y_pred=y_pred, class_names=class_names)
    print("Confusion Matrix:")
    print(matrix)
    print("Classification Report")
    print(classification_report(y_true=y_true, y_pred=y_pred, zero_division=0))


def save_classification_report(
    y_true: List[str], y_pred: List[str], results_dir: Path, class_names: List[str]
) -> None:
    matrix = _conf_mat(y_true=y_true, y_pred=y_pred, class_names=class_names)
    np.savetxt(f"{results_dir}/confusion_matrix.csv", matrix, delimiter=",")
    pd.DataFrame(matrix, index=class_names, columns=class_names).to_csv(
        f"{results_dir}/confusion_matrix_headers.csv", index=True
    )
    report = classification_report(y_true, y_pred, output_dict=True, zero_division=0)
    pd.DataFrame.from_dict(report, orient="columns", dtype=None, columns=None).to_csv(
        f"{results_dir}/classification_metrics.csv", index=True
    )


def _get_ordered_class_names(train_generator: DirectoryIterator) -> List[str]:
    labels_dict = train_generator.class_indices
    n_classes = len(labels_dict)
    return [next(k for k, v in labels_dict.items() if v == i) for i in range(n_classes)]


def predictions_with_truths(
    model: tf.keras.Model, validation_generator: DirectoryIterator
) -> Tuple[List[str], List[str]]:
    predictions: List[str] = []
    truths: List[str] = []
    label_number_to_name_map = {
        number: name for name, number in validation_generator.class_indices.items()
    }
    for image, label in tqdm(validation_generator):
        # model.predict(image)
        predictions += [
            label_number_to_name_map[label_number]
            for label_number in np.argmax(model.predict(image), axis=1)
        ]
        truths += [
            label_number_to_name_map[label_number]
            for label_number in np.argmax(label, axis=1)
        ]
        if len(predictions) >= 4:  # len(validation_generator.classes):
            break
    return predictions, truths
