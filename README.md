# STYLEGAN3 - ImageControl

## Description

With this project you'll be able to generate editing vectors to modify syntetic images.

## Steps
1. Run `gen_images.py` to generate the images and the latent vectors of the amount of images you want.
2. Run `label.py` to label the images. Choose the classifiers depending on the attributtes you want to modify.
3. Run `svm.py` to train the SVMs and generate the editing vectors. Choose the classes depending on the attributes you want to modify.
4. Run `gui/gui.py` to open the GUI and modify the images.