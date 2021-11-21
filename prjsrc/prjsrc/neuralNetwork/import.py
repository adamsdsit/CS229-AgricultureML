import numpy as np
from PIL import Image
import os

def createFileList(myDir, format='.jpg'):
    fileList = []
    print(myDir)
    for root, dirs, files in os.walk(myDir, topdown=False):
        for name in files:
            if name.endswith(format):
                fullName = os.path.join(root, name)
                fileList.append(fullName)
    return fileList

def get_train_labels():
    train_labels = np.loadtxt('train_labels_only.csv', delimiter=',')
    return train_labels

def get_test_labels():
    test_labels = np.loadtxt('test_labels_only.csv', delimiter=',')
    return test_labels

def get_train_data():
    myFileList = createFileList('./train_images/')
    myFileList = sorted(myFileList)
    train_data = np.array([])
    for file in myFileList:
        img_file = Image.open(file)
        if train_data.size == 0:
            train_data = np.asarray(img_file.getdata(), dtype=np.int).reshape(750000,1)
        else:
            temp_array = np.asarray(img_file.getdata(), dtype=np.int).reshape(750000,1)
            train_data = np.append(train_data,temp_array,axis=1)
        print(train_data.T.shape)
    return train_data.T

def get_test_data():
    myFileList = createFileList('./test_images/')
    myFileList = sorted(myFileList)
    test_data = np.array([])
    for file in myFileList:
        img_file = Image.open(file)
        if test_data.size == 0:
            test_data = np.asarray(img_file.getdata(), dtype=np.int).reshape(750000, 1)
        else:
            temp_array = np.asarray(img_file.getdata(), dtype=np.int).reshape(750000, 1)
            test_data = np.append(test_data, temp_array, axis=1)
        print(test_data.T.shape)
    return test_data.T