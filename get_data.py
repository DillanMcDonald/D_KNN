"""
File: get_data.py

Author(s): Kayvon Khosrowpour
Date created: 10-19-18

Description:
Provides functions to get the data from the CIFAR-100 dataset and demos
how to use the functions.
> my_knn
    > cifair-100
        - meta
        - test
        - train
    - knn.py
"""

import os
import numpy as np
from matplotlib import pyplot as plt

img_dataset_train_path = os.path.normpath('cifar-100/train')
img_dataset_test_path = os.path.normpath('cifar-100/test')

CIFAR100_LABELS_LIST = [
    'apple', 'aquarium_fish', 'baby', 'bear', 'beaver', 'bed', 'bee', 'beetle', 
    'bicycle', 'bottle', 'bowl', 'boy', 'bridge', 'bus', 'butterfly', 'camel', 
    'can', 'castle', 'caterpillar', 'cattle', 'chair', 'chimpanzee', 'clock', 
    'cloud', 'cockroach', 'couch', 'crab', 'crocodile', 'cup', 'dinosaur', 
    'dolphin', 'elephant', 'flatfish', 'forest', 'fox', 'girl', 'hamster', 
    'house', 'kangaroo', 'keyboard', 'lamp', 'lawn_mower', 'leopard', 'lion',
    'lizard', 'lobster', 'man', 'maple_tree', 'motorcycle', 'mountain', 'mouse',
    'mushroom', 'oak_tree', 'orange', 'orchid', 'otter', 'palm_tree', 'pear',
    'pickup_truck', 'pine_tree', 'plain', 'plate', 'poppy', 'porcupine',
    'possum', 'rabbit', 'raccoon', 'ray', 'road', 'rocket', 'rose',
    'sea', 'seal', 'shark', 'shrew', 'skunk', 'skyscraper', 'snail', 'snake',
    'spider', 'squirrel', 'streetcar', 'sunflower', 'sweet_pepper', 'table',
    'tank', 'telephone', 'television', 'tiger', 'tractor', 'train', 'trout',
    'tulip', 'turtle', 'wardrobe', 'whale', 'willow_tree', 'wolf', 'woman',
    'worm'
]
def display_img(name, img, show=True, cmap='gray'):
    """
    Simple function to display an RGB image in matplotlib.
    Input
        name: the title of the img to show
        img: the rgb image
        show (optional): if True, will display it. If False, just adds it
            to the queue. Eventually plt.show() must be called.
        cmap (optional): the cmap to display the image in. If None, will
            display as rgb.
    """

    figure = plt.figure()
    axes = plt.axes()
    axes.set_title(name)
    axes.imshow(img, cmap=cmap)
    if show:
        plt.show()

def unpickle(file):
    """
    Unpickles the CIFAR stores. Provided by their website.
    """
    import pickle
    with open(file, 'rb') as fo:
        dict = pickle.load(fo, encoding='bytes')
    return dict

def get_data(data_path, as_str=True):
    """
    Gets the training data from the data set. Returns the images and labels as
    numpy arrays.
    Input:
        data_path: the string path to the data file (i.e. train or test).
        as_str: if True, will also return the labels by their string representations
    Output:
        imgs: the training images (each row is an image)
        labels: the labels of the images as integers
    """
    dataset = unpickle(data_path)
    imgs = np.array(dataset[b'data'], dtype=np.uint8)
    labels = np.array(dataset[b'fine_labels'], dtype=np.uint8).reshape(-1, 1)
    
    if as_str:
        train_lbls_str = np.array([CIFAR100_LABELS_LIST[i[0]] for i in labels])
        return imgs, labels, train_lbls_str

    return imgs, labels

def cifar_to_rgb(cifar_img):
    """
    Given a cifar image, reshapes it to be an rgb image.
    """
    r = cifar_img[:1024].reshape(32, 32)
    g = cifar_img[1024:2048].reshape(32, 32)
    b = cifar_img[2048:]. reshape(32, 32)
    rgb = np.dstack([r, g, b])
    return rgb

# example of how to use the get function
#test_imgs, test_lbls, test_lbls_str = get_data(img_dataset_test_path)
#print(test_imgs.shape, test_lbls.shape)
#for i in range(0, 10):
#    print(test_lbls[i], test_lbls_str[i])
#print()

# display some examples
#for j in range(10, 20):
#    img = cifar_to_rgb(test_imgs[j])
#    display_img(test_lbls_str[j], img, show=False, cmap=None)

#plt.show()

