## Grad-CAM implementation in TensorFlow ##

Original paper: https://openaccess.thecvf.com/content_ICCV_2017/papers/Selvaraju_Grad-CAM_Visual_Explanations_ICCV_2017_paper.pdf

This implementation is based on https://keras.io/examples/vision/grad_cam/

Work in progress. The current version uses VGG16 with ImageNet weights.

#### Rerquirements ####

python 3.7 (later versions of python are not supported by TensorFlow at the time of writing)

#### Usage ####

python src/main.py examples/

The processed images will be placed in the "predictions/examples" folder.

### Data ###

A reasonable 6-class dataset of bridge defects:
("MultiClassifer" folder only, ignore the rest)

https://zenodo.org/record/2601506#.YBhsuZdxcuU

This repository is published under the MIT license.
