import tensorflow as tf
import numpy as np
from sklearn.model_selection import train_test_split
from tensorflow.keras.layers import Input, Add, Dense, Activation, ZeroPadding2D, BatchNormalization, Flatten, Conv2D, AveragePooling2D, MaxPooling2D, GlobalMaxPooling2D
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.initializers import RandomUniform, glorot_uniform, constant, identity
from PIL import Image
import os
import pandas as pd
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, confusion_matrix
from sklearn.metrics import roc_curve
from sklearn.metrics import ConfusionMatrixDisplay
import matplotlib.pyplot as plt

def identity_block(X, f, filters, training=True, initializer=RandomUniform):
    """
    Implementation of the identity block as defined in Figure 4
    
    Arguments:
    X -- input tensor of shape (m, n_H_prev, n_W_prev, n_C_prev)
    f -- integer, specifying the shape of the middle CONV's window for the main path
    filters -- python list of integers, defining the number of filters in the CONV layers of the main path
    training -- True: Behave in training mode
                False: Behave in inference mode
    initializer -- to set up the initial weights of a layer. Equals to random uniform initializer
    
    Returns:
    X -- output of the identity block, tensor of shape (m, n_H, n_W, n_C)
    """
    
    # Retrieve Filters
    F1, F2, F3 = filters
    
    # Save the input value. You'll need this later to add back to the main path. 
    X_shortcut = X
    
    # First component of main path
    X = Conv2D(filters = F1, kernel_size = 1, strides = (1,1), padding = 'valid', kernel_initializer = initializer(seed=0))(X)
    X = BatchNormalization(axis = 3)(X, training = training) # Default axis
    X = Activation('relu')(X)
    
    ### START CODE HERE
    ## Second component of main path (≈3 lines)
    X = Conv2D(filters = F2, kernel_size = f, strides = (1,1), padding = 'same', kernel_initializer = initializer(seed=0))(X)
    X = BatchNormalization(axis = 3)(X, training = training)
    X = Activation('relu')(X)

    ## Third component of main path (≈2 lines)
    X = Conv2D(filters = F3, kernel_size = 1, strides = (1,1), padding = 'valid', kernel_initializer = initializer(seed=0))(X)
    X = BatchNormalization(axis = 3)(X, training = training) 
    
    ## Final step: Add shortcut value to main path, and pass it through a RELU activation (≈2 lines)
    X = Add()([X, X_shortcut])
    X = Activation('relu')(X) 
    ### END CODE HERE

    return X

def convolutional_block(X, f, filters, s = 2, training=True, initializer=glorot_uniform):
    """
    Implementation of the convolutional block as defined in Figure 4
    
    Arguments:
    X -- input tensor of shape (m, n_H_prev, n_W_prev, n_C_prev)
    f -- integer, specifying the shape of the middle CONV's window for the main path
    filters -- python list of integers, defining the number of filters in the CONV layers of the main path
    s -- Integer, specifying the stride to be used
    training -- True: Behave in training mode
                False: Behave in inference mode
    initializer -- to set up the initial weights of a layer. Equals to Glorot uniform initializer, 
                   also called Xavier uniform initializer.
    
    Returns:
    X -- output of the convolutional block, tensor of shape (n_H, n_W, n_C)
    """
    
    # Retrieve Filters
    F1, F2, F3 = filters
    
    # Save the input value
    X_shortcut = X

    ##### MAIN PATH #####
    
    # First component of main path glorot_uniform(seed=0)
    X = Conv2D(filters = F1, kernel_size = 1, strides = (s, s), padding='valid', kernel_initializer = initializer(seed=0))(X)
    X = BatchNormalization(axis = 3)(X, training=training)
    X = Activation('relu')(X)

    ### START CODE HERE
    
    ## Second component of main path (≈3 lines)
    X = Conv2D(filters = F2, kernel_size = f, strides = (1, 1), padding='same', kernel_initializer = glorot_uniform(seed=0))(X)
    X = BatchNormalization(axis = 3)(X, training=training) 
    X = Activation('relu')(X) 

    ## Third component of main path (≈2 lines)
    X = Conv2D(filters = F3, kernel_size = 1, strides = (1, 1), padding='valid', kernel_initializer = glorot_uniform(seed=0))(X)
    X = BatchNormalization(axis = 3)(X, training=training)  
    
    ##### SHORTCUT PATH ##### (≈2 lines)
    X_shortcut = Conv2D(filters=F3, kernel_size = 1, strides = (s, s), padding='valid', kernel_initializer = glorot_uniform(seed=0))(X_shortcut)
    X_shortcut = BatchNormalization(axis = 3)(X_shortcut, training=training)
    
    ### END CODE HERE

    # Final step: Add shortcut value to main path (Use this order [X, X_shortcut]), and pass it through a RELU activation
    X = Add()([X, X_shortcut])
    X = Activation('relu')(X)
    
    return X

def ResNet50(input_shape = (64 , 64, 3), classes = 7):
    """
    Stage-wise implementation of the architecture of the popular ResNet50:
    CONV2D -> BATCHNORM -> RELU -> MAXPOOL -> CONVBLOCK -> IDBLOCK*2 -> CONVBLOCK -> IDBLOCK*3
    -> CONVBLOCK -> IDBLOCK*5 -> CONVBLOCK -> IDBLOCK*2 -> AVGPOOL -> FLATTEN -> DENSE 

    Arguments:
    input_shape -- shape of the images of the dataset
    classes -- integer, number of classes

    Returns:
    model -- a Model() instance in Keras
    """
    
    # Define the input as a tensor with shape input_shape
    X_input = Input(input_shape)

    # Zero-Padding
    X = ZeroPadding2D((3, 3))(X_input)
    
    # Stage 1
    X = Conv2D(64, (7, 7), strides = (2, 2), kernel_initializer = glorot_uniform(seed=0))(X)
    X = BatchNormalization(axis = 3)(X)
    X = Activation('relu')(X)
    X = MaxPooling2D((3, 3), strides=(2, 2))(X)

    # Stage 2
    X = convolutional_block(X, f = 3, filters = [64, 64, 256], s = 1)
    X = identity_block(X, 3, [64, 64, 256])
    X = identity_block(X, 3, [64, 64, 256])
    
    ## Stage 3 (≈4 lines)
    X = convolutional_block(X, f=3, filters = [128, 128, 512], s=2) 
    X = identity_block(X, 3, [128, 128, 512]) 
    X = identity_block(X, 3, [128, 128, 512])
    X = identity_block(X, 3, [128, 128, 512]) 
    
    ## Stage 4 (≈6 lines)
    X = convolutional_block(X, f=3, filters = [256, 256, 1024], s=2) 
    X = identity_block(X, 3, [256, 256, 1024]) 
    X = identity_block(X, 3, [256, 256, 1024]) 
    X = identity_block(X, 3, [256, 256, 1024]) 
    X = identity_block(X, 3, [256, 256, 1024]) 
    X = identity_block(X, 3, [256, 256, 1024]) 

    ## Stage 5 (≈3 lines)
    X = convolutional_block(X, f=3, filters = [512, 512, 2048], s=2) 
    X = identity_block(X, 3, [512, 512, 2048]) 
    X = identity_block(X, 3, [512, 512, 2048]) 

    ## AVGPOOL (≈1 line). Use "X = AveragePooling2D(...)(X)"
    X = AveragePooling2D(pool_size=(2, 2), padding='same')(X)

    # output layer
    X = Flatten()(X)
    X = Dense(classes, activation='softmax', kernel_initializer = glorot_uniform(seed=0))(X)
    
    # Create model
    model = Model(inputs = X_input, outputs = X)

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
            img_array = np.asarray(img_file.getdata(), dtype=np.int)
            # img_array += np.array([-30, 0, 30])
            # img_augmented = fancy_pca.fancy_pca(asarray(img_file))
            train_data = img_array.reshape(1, 500, 500, 3)
        else:
            img_array = np.asarray(img_file.getdata(), dtype=np.int)
            # img_array += np.array([-30, 0, 30])
            img_array = img_array.reshape(1, 500, 500, 3)
            # img_augmented = fancy_pca.fancy_pca(img_array)
            train_data = np.append(train_data,img_array,axis=0)
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
    # print(edge_val)
    # df.plot.hist(bins=7, rwidth=1, edgecolor='black')
    # plt.show()
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

def process_images(image, label):
    # Normalize images to have a mean of 0 and standard deviation of 1
    image = tf.image.per_image_standardization(image)
    # Resize images from 500x500 to 64x64
    image = tf.image.resize(image, (64,64))
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

# For the purposes of this assignment, you've been asked to train the model for ten epochs. You can see that it performs well. The online grader will only run your code for a small number of epochs as well. Please go ahead and submit your assignment.

# After you have finished this official (graded) part of this assignment, you can also optionally train the ResNet for more iterations, if you want. It tends to get much better performance when trained for ~20 epochs, but this does take more than an hour when training on a CPU.
#
# Using a GPU, this ResNet50 model's weights were trained on the SIGNS dataset. You can load and run the trained model on the test set in the cells below. It may take ≈1min to load the model. Have fun!

# pre_trained_model = tf.keras.models.load_model('resnet50.h5')
# preds = pre_trained_model.evaluate(X_test, Y_test)

def main(plot=True):
    # Loading the data (signs)
    orig_x, orig_y = read_data()
    orig_y = classification(orig_y, orig_y)
    orig_y = one_hot_labels(orig_y)
    orig_x = orig_x / 255.
    # Create the training sample
    train_x, val_x, train_y, val_y = train_test_split(orig_x, orig_y, test_size=0.3, random_state=1)
    # Split the remaining observations into validation and test
    # val_x, test_x, val_y, test_y = train_test_split(val_test_x, val_test_y, test_size=0.5, random_state=1)

    print("number of training examples = " + str(train_x.shape[0]))
    print("number of validation examples = " + str(val_x.shape[0]))
    print("X_train shape: " + str(train_x.shape))
    print("Y_train shape: " + str(train_y.shape))
    # print("X_test shape: " + str(val_x.shape))
    # print("Y_test shape: " + str(val_y.shape))

    model = ResNet50(input_shape=(64, 64, 3), classes=7)
    lr_schedule = tf.optimizers.schedules.ExponentialDecay(
        initial_learning_rate=4e-2,
        decay_steps=640,
        decay_rate=0.5
    )
    lr_reduce = tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=1, min_lr=0.00001)
    adam = tf.optimizers.Adam(learning_rate=0.01)
    sgd = tf.optimizers.SGD(learning_rate=0.1, momentum=0.95, nesterov=True, decay=1e-06)
    model.compile(optimizer=sgd,
                       loss='categorical_crossentropy',
                       metrics=['accuracy'])
    model.summary()

    datagen = ImageDataGenerator(featurewise_std_normalization=True)
    train_batches = datagen.flow(train_x, train_y, batch_size=64)
    train_crops = crop_generator(train_batches, 64)
    val_batches = datagen.flow(val_x, val_y, batch_size=64)
    val_crops = crop_generator(val_batches, 64)
    test_batches = datagen.flow(val_x, val_y, batch_size=1, shuffle=False)
    test_crops = crop_generator(test_batches, 64)

    # history = conv_model.fit(train_ds, epochs=100, validation_data=test_ds)
    history = model.fit_generator(train_crops, epochs=100, steps_per_epoch=64,
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
    x = np.empty((1000, 227, 227, 3))
    for i in range(1000):
        a, b = test_crops.__next__()
        temp_x.extend(a)
        x[i,] = a
        y.extend(b)
    print(x.shape)
    Pred_y = model.predict(x)
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

    scores = model.evaluate(test_crops, steps=1000)
    print(scores)

if __name__ == '__main__':
    main()