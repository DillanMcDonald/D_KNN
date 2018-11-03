"""
File: Dillan_KNN.py
Author(s): Dillan McDonald
Date created: 11-3-18

Description:
Runs a KNN for the CIFAR-100 dataset at varying k values from 1 to 20
"""
import get_data as gd
import numpy as np
import knn_algo as knnalgo
from matplotlib import pyplot as plt

train_imgs, train_lbls, train_lbls_str = gd.get_data(gd.img_dataset_train_path)
test_imgs, test_lbls, test_lbls_str = gd.get_data(gd.img_dataset_test_path)
for j in range(1,20):
    knn = knnalgo.KNN(j) # 3 Nearest neighbour
    knn.train(train_imgs, train_lbls)
    data_point_no = 100;
    sample_test_data = test_imgs[:data_point_no, :]

    pred = knn.predict(sample_test_data)
    ac = 0
    for i in range(0, len(pred)):
        #img = gd.cifar_to_rgb(test_imgs[i])
        #gd.display_img(train_lbls_str[pred[i]], img, show=False, cmap=None)
        if train_lbls_str[pred[i]] == test_lbls_str[i]:
            ac = ac + 1
    ac = ac/data_point_no

    print("Accuracy (%): ", ac*100, " K value: ",j)
#for i in range(0, 10):
#    print(test_lbls[i], test_lbls_str[i])
#print()

#display some examples
#for j in range(10, 20):
#    img = gd.cifar_to_rgb(test_imgs[j])
#    gd.display_img(test_lbls_str[j], img, show=False, cmap=None)

#plt.show()
