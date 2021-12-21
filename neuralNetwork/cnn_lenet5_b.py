import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import pandas as pd
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import os
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, confusion_matrix
from sklearn.metrics import roc_curve
from sklearn.metrics import ConfusionMatrixDisplay


def one_hot_labels(labels):
    """
    Convert labels into ont-hot representation
    Args:
        labels: input labels

    Returns:
        one-hot representation of the labels
    """
    one_hot_labels = np.zeros((labels.size, 7))
    one_hot_labels[np.arange(labels.size),labels.astype(int)] = 1
    return one_hot_labels

def convolutional_model(input_shape):
    """
    Implements the forward propagation for the model:
    CONV2D -> RELU -> MAXPOOL -> CONV2D -> RELU -> MAXPOOL -> FLATTEN -> DENSE

    Arguments:
    input_img -- input dataset, of shape (input_shape)

    Returns:
    model -- TF Keras model (object containing the information for the entire training process)
    """
    input_img = tf.keras.Input(shape=input_shape)
    # YOUR CODE STARTS HERE
    ## CONV2D: 8 filters 4x4, stride of 1, padding 'SAME'
    Z1 = tf.keras.layers.Conv2D(8, (4, 4), strides=(1, 1), padding='SAME')(input_img)
    ## RELU
    A1 = tf.keras.layers.ReLU()(Z1)
    ## MAXPOOL: window 8x8, stride 8, padding 'SAME'
    P1 = tf.keras.layers.MaxPooling2D(pool_size=(8, 8), strides=(8, 8), padding='SAME')(A1)
    ## CONV2D: 16 filters 2x2, stride 1, padding 'SAME'
    Z2 = tf.keras.layers.Conv2D(16, (2, 2), strides=(1, 1), padding='SAME')(P1)
    ## RELU
    A2 = tf.keras.layers.ReLU()(Z2)
    ## MAXPOOL: window 4x4, stride 4, padding 'SAME'
    P2 = tf.keras.layers.MaxPooling2D(pool_size=(4, 4), strides=(4, 4), padding='SAME')(A2)
    ## FLATTEN
    F = tf.keras.layers.Flatten()(P2)
    ## Dense layer
    ## 6 neurons in output layer. Hint: one of the arguments should be "activation='softmax'"
    outputs = tf.keras.layers.Dense(7, activation='softmax')(F)

    # YOUR CODE ENDS HERE
    model = tf.keras.Model(inputs=input_img, outputs=outputs)
    return model

def read_data():
    """
    Get data from images folder (X) and from CSV (labels)
    Returns:
        Array of original set
    """
    x = get_data()
    y = get_labels()
    return x, y

def createFileList(myDir, format='.jpg'):
    """
    Create a file list based on format given.
    Args:
        myDir: the folder to build the list.
        format: file format to consider.

    Returns:
        A list of all files in the specified folder with speficied format.
    """
    fileList = []
    print(myDir)
    for root, dirs, files in os.walk(myDir, topdown=False):
        for name in files:
            if name.endswith(format):
                fullName = os.path.join(root, name)
                fileList.append(fullName)
    return fileList

def get_data():
    """
    Get the images - RGB values from all examples.
    Returns:
        Array of RGB values.
    """
    path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'images'))
    myFileList = createFileList(path)
    myFileList = sorted(myFileList)
    train_data = np.array([])
    for file in myFileList:
        img_file = Image.open(file)
        if train_data.size == 0:
            train_data = np.asarray(img_file.getdata(), dtype=np.int).reshape(1, 500, 500, 3)
        else:
            temp_array = np.asarray(img_file.getdata(), dtype=np.int).reshape(1, 500, 500, 3)
            train_data = np.append(train_data,temp_array,axis=0)
        print(train_data.shape)
    return train_data

def get_labels():
    """
    Read labels from CSV (C-Level real values).
    Returns:
        A vector of labels.
    """
    train_labels = np.loadtxt('labels.csv', delimiter=',')
    return train_labels

def classification(y_real, y_pred):
    """ Transforms y_pred on buckets based on y_real values.

    Returns:
        Numpy array of classified labels (n_examples,).
    """
    edge_val = np.histogram_bin_edges(y_real, bins=7, range=None, weights=None)
    df = pd.DataFrame(y_real)
    print(edge_val)
    df.plot.hist(bins=7, rwidth=1, edgecolor='black')
    plt.show()
    categories = np.empty(y_pred.shape)
    for i in range(categories.size):
        if y_pred[i] < edge_val[1]:
            categories[i] = 0
        elif y_pred[i] >= edge_val[1] and y_pred[i] < edge_val[2]:
            categories[i] = 1
        elif y_pred[i] >= edge_val[2] and y_pred[i] < edge_val[3]:
            categories[i] = 2
        elif y_pred[i] >= edge_val[3] and y_pred[i] < edge_val[4]:
            categories[i] = 3
        elif y_pred[i] >= edge_val[4] and y_pred[i] < edge_val[5]:
            categories[i] = 4
        elif y_pred[i] >= edge_val[5] and y_pred[i] < edge_val[6]:
            categories[i] = 5
        else:
            categories[i] = 6
    return categories

def process_images(image, label):
    # Normalize images to have a mean of 0 and standard deviation of 1
    image = tf.image.per_image_standardization(image)
    # Resize images from 500x500 to 256x256
    image = tf.image.resize(image, (256,256))
    return image, label

def random_crop(img, random_crop_size):
    assert img.shape[2] == 3
    height, width = img.shape[0], img.shape[1]
    dy, dx = random_crop_size
    x = np.random.randint(0, width - dx + 1)
    y = np.random.randint(0, height - dy + 1)
    return img[y:(y+dy), x:(x+dx), :]

def crop_generator(batches, crop_length):
    while True:
        batch_x, batch_y = next(batches)
        batch_crops = np.zeros((batch_x.shape[0], crop_length, crop_length, 3))
        for i in range(batch_x.shape[0]):
            batch_crops[i] = random_crop(batch_x[i], (crop_length, crop_length))
        yield (batch_crops, batch_y)

def main(plot=True):
    # y = get_labels()
    # y = classification(y, y)
    # np.savetxt("y.csv", y, delimiter=",")
    # quit()
    # test = np.array([[76,  49,  12,   8,  12,   0,   0],
    #                  [45, 115,  19,  24,   9,   1,   0],
    #                  [16, 23, 22, 53, 16, 11, 0],
    #                  [6,  14,  50, 142,  31,   2,   1],
    #                  [11, 6, 14, 48, 116, 10, 2],
    #                  [0,   0,   0,   0,   0,  12,   0],
    #                  [0, 18, 0, 0, 5, 1, 0]])
    # cmd = ConfusionMatrixDisplay(confusion_matrix=test)
    # font = {'family': 'Arial',
    #         'weight': 'bold',
    #         'size': 20}
    # plt.rc('font', **font)
    # cmd.plot()
    # plt.savefig('CNN_Confusion', dpi=300)
    # quit()
    # roc curve for classes
    fpr = {}
    tpr = {}
    thresh = {}
    test_y = np.array([0,1,3,3,3,4,1,3,2,3,3,1,2,4,2,4,2,4,6,3,2,0,4,1,3,1,0,1,3,3,4,3,0,4,4,4,2
                          ,3,2,1,3,2,3,3,4,2,3,0,0,0,3,2,3,4,0,4,3,3,4,1,3,1,4,0,0,0,1,4,0,6,0,4,3,0
                          ,2,3,1,4,4,1,3,0,1,3,3,0,1,6,3,1,3,4,2,1,3,1,6,4,4,4,3,1,5,1,1,4,1,3,2,1,2
                          ,4,3,0,3,4,4,4,4,4,5,2,4,2,1,1,2,2,1,4,1,3,1,0,1,2,0,3,0,0,4,3,3,1,3,2,4,4
                          ,0,2,1,0,3,2,0,3,1,1,0,1,0,2,1,1,0,1,3,3,3,4,1,3,2,3,3,1,2,4,2,4,2,4,6,3,2
                          ,0,4,1,3,1,0,1,3,3,4,3,0,4,4,4,2,3,2,1,3,2,3,3,4,2,3,0,0,0,3,2,3,4,0,4,3,3
                          ,4,1,3,1,4,0,0,0,1,4,0,6,0,4,3,0,2,3,1,4,4,1,3,0,1,3,3,0,1,6,3,1,3,4,2,1,3
                          ,1,6,4,4,4,3,1,5,1,1,4,1,3,2,1,2,4,3,0,3,4,4,4,4,4,5,2,4,2,1,1,2,2,1,4,1,3
                          ,1,0,1,2,0,3,0,0,4,3,3,1,3,2,4,4,0,2,1,0,3,2,0,3,1,1,0,1,0,2,1,1,0,1,3,3,3
                          ,4,1,3,2,3,3,1,2,4,2,4,2,4,6,3,2,0,4,1,3,1,0,1,3,3,4,3,0,4,4,4,2,3,2,1,3,2
                          ,3,3,4,2,3,0,0,0,3,2,3,4,0,4,3,3,4,1,3,1,4,0,0,0,1,4,0,6,0,4,3,0,2,3,1,4,4
                          ,1,3,0,1,3,3,0,1,6,3,1,3,4,2,1,3,1,6,4,4,4,3,1,5,1,1,4,1,3,2,1,2,4,3,0,3,4
                          ,4,4,4,4,5,2,4,2,1,1,2,2,1,4,1,3,1,0,1,2,0,3,0,0,4,3,3,1,3,2,4,4,0,2,1,0,3
                          ,2,0,3,1,1,0,1,0,2,1,1,0,1,3,3,3,4,1,3,2,3,3,1,2,4,2,4,2,4,6,3,2,0,4,1,3,1
                          ,0,1,3,3,4,3,0,4,4,4,2,3,2,1,3,2,3,3,4,2,3,0,0,0,3,2,3,4,0,4,3,3,4,1,3,1,4
                          ,0,0,0,1,4,0,6,0,4,3,0,2,3,1,4,4,1,3,0,1,3,3,0,1,6,3,1,3,4,2,1,3,1,6,4,4,4
                          ,3,1,5,1,1,4,1,3,2,1,2,4,3,0,3,4,4,4,4,4,5,2,4,2,1,1,2,2,1,4,1,3,1,0,1,2,0
                          ,3,0,0,4,3,3,1,3,2,4,4,0,2,1,0,3,2,0,3,1,1,0,1,0,2,1,1,0,1,3,3,3,4,1,3,2,3
                          ,3,1,2,4,2,4,2,4,6,3,2,0,4,1,3,1,0,1,3,3,4,3,0,4,4,4,2,3,2,1,3,2,3,3,4,2,3
                          ,0,0,0,3,2,3,4,0,4,3,3,4,1,3,1,4,0,0,0,1,4,0,6,0,4,3,0,2,3,1,4,4,1,3,0,1,3
                          ,3,0,1,6,3,1,3,4,2,1,3,1,6,4,4,4,3,1,5,1,1,4,1,3,2,1,2,4,3,0,3,4,4,4,4,4,5
                          ,2,4,2,1,1,2,2,1,4,1,3,1,0,1,2,0,3,0,0,4,3,3,1,3,2,4,4,0,2,1,0,3,2,0,3,1,1
                          ,0,1,0,2,1,1,0,1,3,3,3,4,1,3,2,3,3,1,2,4,2,4,2,4,6,3,2,0,4,1,3,1,0,1,3,3,4
                          ,3,0,4,4,4,2,3,2,1,3,2,3,3,4,2,3,0,0,0,3,2,3,4,0,4,3,3,4,1,3,1,4,0,0,0,1,4
                          ,0,6,0,4,3,0,2,3,1,4,4,1,3,0,1,3,3,0,1,6,3,1,3,4,2,1,3,1,6,4,4,4,3,1,5,1,1
                          ,4,1,3,2,1,2,4,3,0,3,4,4,4,4,4,5,2,4,2,1,1,2,2,1,4,1,3,1,0,1,2,0,3,0,0,4,3
                          ,3,1,3,2,4,4,0,2,1,0,3,2,0,3,1,1,0,1,0,2,1,1,0,1,3,3,3,4,1,3,2,3,3,1,2,4,2
                          ,4])
    pred_y = np.array([1,1,1,2,4,4,1,3,3,4,2,0,4,3,3,0,3,1,4,3,0,2,4,1,3,1,0,1,3,3,4,4,0,3,2,4,5
                          ,1,2,3,2,1,2,1,4,3,3,1,0,1,2,3,4,4,0,4,3,0,3,4,4,0,3,0,0,2,1,4,1,1,1,4,3,1
                          ,3,3,1,4,4,1,3,1,0,3,4,0,1,1,3,1,3,3,3,0,3,1,1,0,3,1,3,0,5,4,1,4,0,3,3,2,2
                          ,4,6,1,4,3,4,4,3,5,5,1,4,2,0,1,2,4,1,4,1,3,0,0,3,1,1,3,1,4,4,0,3,1,3,3,5,4
                          ,1,1,1,4,2,3,0,2,0,1,0,1,3,3,4,1,0,1,1,3,3,4,3,3,2,4,4,2,1,2,2,0,3,4,4,3,0
                          ,2,5,1,4,4,0,1,3,2,4,4,0,3,4,4,5,1,5,3,2,0,2,2,4,4,2,1,2,1,2,1,2,3,0,4,3,3
                          ,3,1,3,0,3,4,0,0,1,4,1,1,1,2,3,1,3,3,1,4,3,1,3,0,3,3,1,0,1,1,3,1,3,3,3,0,3
                          ,1,1,0,3,0,3,0,5,1,1,4,3,2,4,2,5,4,3,0,3,2,4,4,4,6,5,1,4,2,0,1,0,4,1,4,2,3
                          ,1,0,3,0,1,3,3,4,4,3,3,1,3,3,2,4,1,1,1,4,2,3,0,3,0,1,0,0,3,3,4,3,0,1,3,3,4
                          ,4,1,3,0,4,2,2,2,3,2,3,5,4,4,3,0,0,6,1,3,4,2,1,3,2,4,4,0,3,4,4,5,1,5,3,3,2
                          ,2,2,4,3,3,1,2,1,3,1,4,4,0,4,3,0,3,1,3,0,3,0,0,0,1,4,1,1,1,4,3,1,3,2,1,4,3
                          ,1,3,0,0,3,4,0,1,1,3,1,3,3,0,0,3,1,1,2,3,0,3,0,5,1,1,4,0,3,4,0,3,4,3,2,3,3
                          ,4,4,4,5,5,1,3,4,0,1,0,0,1,4,1,4,1,0,3,1,1,3,1,4,4,0,3,1,3,3,5,4,1,1,1,2,2
                          ,3,0,3,2,1,0,0,3,3,3,3,0,2,2,2,2,4,3,3,2,4,2,0,4,2,3,2,3,4,1,3,0,0,4,2,3,1
                          ,0,1,3,2,4,2,0,4,4,4,5,1,2,3,3,4,2,1,4,3,3,1,0,0,5,3,2,4,0,4,3,3,3,5,3,0,3
                          ,1,0,0,1,4,1,1,1,4,3,0,3,3,1,4,3,1,3,4,2,3,1,1,1,1,2,1,2,4,3,0,3,1,1,0,3,0
                          ,3,0,5,1,1,4,0,3,4,2,2,4,3,0,3,3,4,4,4,5,5,1,3,2,0,1,0,2,1,4,1,0,0,0,3,1,3
                          ,3,1,4,4,2,3,1,3,3,4,4,1,1,1,3,2,3,0,3,0,1,0,0,0,3,4,1,0,1,2,3,2,4,2,3,3,4
                          ,2,0,4,2,3,0,3,1,4,3,0,0,3,2,4,1,0,1,3,3,4,4,0,3,4,2,5,1,5,3,3,1,2,3,4,3,3
                          ,1,2,0,3,3,4,4,1,4,3,3,3,1,4,0,3,0,0,2,1,4,1,1,1,4,3,1,3,3,1,4,3,1,3,0,3,3
                          ,1,1,1,1,3,1,2,3,4,0,3,1,1,2,3,1,3,1,5,4,2,4,0,3,4,2,1,4,2,0,3,3,4,4,4,5,5
                          ,1,3,2,0,1,0,4,1,4,1,3,1,0,3,0,3,3,1,4,1,3,3,1,3,3,5,4,1,1,1,4,2,3,0,3,0,1
                          ,0,1,3,3,3,1,0,3,2,4,2,4,3,3,3,4,2,3,2,2,3,0,3,4,4,3,0,0,5,1,4,1,0,1,3,2,4
                          ,4,0,4,4,4,5,1,2,3,3,4,2,2,4,2,3,1,2,1,3,1,2,4,0,4,3,3,3,1,3,2,3,0,0,2,1,4
                          ,0,1,1,4,3,1,3,3,1,4,3,1,3,0,2,3,1,1,1,1,3,1,3,3,2,0,3,1,5,0,3,1,3,0,5,1,1
                          ,4,0,3,4,2,1,4,3,0,3,3,4,4,4,5,5,1,3,2,0,1,3,3,1,4,0,3,1,0,3,3,1,3,1,4,4,0
                          ,3,1,3,3,2,4,1,1,1,4,5,3,0,2,0,1,0,0,0,3,4,1,0,2,3,4,2,4,1,4,3,4,4,2,2,3,3
                          ,2])
    n_class = 7
    for i in range(n_class):
        fpr[i], tpr[i], thresh[i] = roc_curve(test_y, pred_y, pos_label=i)

    # plotting
    plt.plot(fpr[0], tpr[0], linestyle='--', color='orange', label='Class 0 vs Rest')
    plt.plot(fpr[1], tpr[1], linestyle='--', color='green', label='Class 1 vs Rest')
    plt.plot(fpr[2], tpr[2], linestyle='--', color='blue', label='Class 2 vs Rest')
    plt.plot(fpr[3], tpr[3], linestyle='--', color='magenta', label='Class 3 vs Rest')
    plt.plot(fpr[4], tpr[4], linestyle='--', color='cyan', label='Class 4 vs Rest')
    plt.plot(fpr[5], tpr[5], linestyle='--', color='brown', label='Class 5 vs Rest')
    plt.plot(fpr[6], tpr[6], linestyle='--', color='red', label='Class 6 vs Rest')
    plt.title('AlexNet ROC curve')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive rate')
    plt.legend(loc='best')
    plt.savefig('AlexNet_ROC', dpi=300)
    quit()
    # Loading the data (signs)
    orig_x, orig_y = read_data()
    orig_y = classification(orig_y, orig_y)
    orig_y = one_hot_labels(orig_y)
    orig_x = orig_x / 255.
    # Create the training sample
    train_x, val_x, train_y, val_y = train_test_split(orig_x, orig_y, test_size=0.3, random_state=1)
    # Split the remaining observations into validation and test
    # val_x, test_x, val_y, test_y = train_test_split(val_test_x, val_test_y, test_size=0.5, random_state=1)

    # Example of an image from the dataset
    index = 9
    plt.imshow(orig_x[index])

    print("number of training examples = " + str(train_x.shape[0]))
    print("number of validation examples = " + str(val_x.shape[0]))
    print("X_train shape: " + str(train_x.shape))
    print("Y_train shape: " + str(train_y.shape))
    # print("X_test shape: " + str(val_x.shape))
    # print("Y_test shape: " + str(val_y.shape))

    conv_model = convolutional_model((256, 256, 3))
    lr_schedule = tf.optimizers.schedules.ExponentialDecay(
        initial_learning_rate=5e-3,
        decay_steps=640,
        decay_rate=0.9
    )
    adam = tf.optimizers.Adam(learning_rate=lr_schedule)
    sgd = tf.optimizers.SGD(learning_rate=lr_schedule)
    conv_model.compile(optimizer=adam,
                       loss='categorical_crossentropy',
                       metrics=['accuracy'])
    conv_model.summary()

    datagen = ImageDataGenerator(featurewise_std_normalization=True)
    train_batches = datagen.flow(train_x, train_y, batch_size=64)
    train_crops = crop_generator(train_batches, 256)
    val_batches = datagen.flow(val_x, val_y, batch_size=64)
    val_crops = crop_generator(val_batches, 256)
    test_batches = datagen.flow(val_x, val_y, batch_size=1, shuffle=False)
    test_crops = crop_generator(test_batches, 256)

    # history = conv_model.fit(train_ds, epochs=100, validation_data=test_ds)
    history = conv_model.fit_generator(train_crops, epochs=100, steps_per_epoch=64,
                                       validation_data=val_crops, validation_steps=64)

    print(history.history)
    df_loss_acc = pd.DataFrame(history.history)
    df_loss = df_loss_acc[['loss', 'val_loss']]
    # df_loss.rename(columns={'loss': 'train', 'val_loss': 'validation'}, inplace=True)
    df_acc = df_loss_acc[['accuracy', 'val_accuracy']]
    # df_acc.rename(columns={'accuracy': 'train', 'val_accuracy': 'validation'}, inplace=True)
    df_loss.plot(title='Model loss', figsize=(12, 8)).set(xlabel='Epoch', ylabel='Loss')
    df_acc.plot(title='Model Accuracy', figsize=(12, 8)).set(xlabel='Epoch', ylabel='Accuracy')
    plt.show()

    # Pred_y = conv_model.predict(test_crops, steps=1000)
    # pred_y = np.argmax(Pred_y, axis=1)
    # print('test_crops')
    temp_x, y = [], []
    x = np.empty((1000, 256, 256, 3))
    for i in range(1000):
        a, b = test_crops.__next__()
        temp_x.extend(a)
        x[i,] = a
        y.extend(b)
    print(x.shape)
    Pred_y = conv_model.predict(x)
    pred_y = np.argmax(Pred_y, axis=1)
    Test_y = np.array(y)
    test_y = np.argmax(Test_y, axis=1)
    print(test_y)
    print('pred_y')
    print(pred_y)
    # Calculate accuracy, precision, recall and confusion matrix
    print('Test Accuracy: ', accuracy_score(test_y, pred_y))
    print('Test Precision: ', precision_score(test_y, pred_y, zero_division=0, average='weighted'))
    print('Test Recall: ', recall_score(test_y, pred_y, average='weighted'))
    print('Test f1', f1_score(test_y, pred_y, zero_division=0, average='weighted', labels=np.unique(pred_y)))
    cm = confusion_matrix(test_y, pred_y)
    print('Confustion Matrix', cm)
    cmd = ConfusionMatrixDisplay(confusion_matrix=cm)
    cmd.plot()
    plt.savefig('AlexNet_Confusion', dpi=300)
    plt.clf()
    n_class = 7

    # roc curve for classes
    fpr = {}
    tpr = {}
    thresh = {}

    for i in range(n_class):
        fpr[i], tpr[i], thresh[i] = roc_curve(test_y, pred_y, pos_label=i)

    # plotting
    plt.plot(fpr[0], tpr[0], linestyle='--', color='orange', label='Class 0 vs Rest')
    plt.plot(fpr[1], tpr[1], linestyle='--', color='green', label='Class 1 vs Rest')
    plt.plot(fpr[2], tpr[2], linestyle='--', color='blue', label='Class 2 vs Rest')
    plt.plot(fpr[3], tpr[3], linestyle='--', color='magenta', label='Class 3 vs Rest')
    plt.plot(fpr[4], tpr[4], linestyle='--', color='cyan', label='Class 4 vs Rest')
    plt.plot(fpr[5], tpr[5], linestyle='--', color='brown', label='Class 5 vs Rest')
    plt.plot(fpr[6], tpr[6], linestyle='--', color='red', label='Class 6 vs Rest')
    plt.title('AlexNet ROC curve')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive rate')
    plt.legend(loc='best')
    plt.savefig('AlexNet_ROC', dpi=300)

    scores = conv_model.evaluate(test_crops, steps=1000)
    print(scores)

if __name__ == '__main__':
    main()