import numpy as np
import matplotlib.pyplot as plt
import argparse
import os
from PIL import Image
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, confusion_matrix
from sklearn.metrics import roc_curve
from sklearn.metrics import ConfusionMatrixDisplay

def softmax(x):
    """
    Compute softmax function for a batch of input values.
    The first dimension of the input corresponds to the batch size. The second dimension
    corresponds to every class in the output.

    Args:
        x: A 2d numpy float array of shape batch_size x number_of_classes

    Returns:
        A 2d numpy float array containing the softmax results of shape batch_size x number_of_classes
    """
    # Calculate the max value for each line of batch size
    max_value = np.max(x, axis=1)[:, np.newaxis]
    # Divide each value by the max of collection: EXP(x)/EXP(max(x)) = EXP(x-max(x))
    x = x - max_value
    # Calculate the EPX(X-max(X))
    exp_x = np.exp(x)
    # Calculate the softmax value for each value, considering the sum only on second dimension
    softmax = exp_x / np.sum(exp_x, axis=1)[:, np.newaxis]
    return softmax

def sigmoid(x):
    """
    Compute the sigmoid function for the input here.

    Args:
        x: A numpy float array

    Returns:
        A numpy float array containing the sigmoid results
    """
    sigmoid = 1 / (1 + np.exp(-x))
    return sigmoid

def get_initial_params(input_size, num_hidden, num_output):
    """
    Compute the initial parameters for the neural network.

    This function should return a dictionary mapping parameter names to numpy arrays containing
    the initial values for those parameters.

    There should be four parameters for this model:
    W1 is the weight matrix for the hidden layer of size input_size x num_hidden
    b1 is the bias vector for the hidden layer of size num_hidden
    W2 is the weight matrix for the output layers of size num_hidden x num_output
    b2 is the bias vector for the output layer of size num_output

    Args:
        input_size: The size of the input data
        num_hidden: The number of hidden states
        num_output: The number of output classes

    Returns:
        A dict mapping parameter names to numpy arrays
    """

    return_dict = {}
    return_dict['W1'] = np.random.normal(size=(input_size, num_hidden))
    return_dict['b1'] = np.zeros(num_hidden)
    return_dict['W2'] = np.random.normal(size=(num_hidden, num_output))
    return_dict['b2'] = np.zeros(num_output)
    return return_dict

def forward_prop(data, labels, params):
    """
    Implement the forward layer given the data, labels, and params.

    Args:
        data: A numpy array containing the input
        labels: A 2d numpy array containing the labels
        params: A dictionary mapping parameter names to numpy arrays with the parameters.
            This numpy array will contain W1, b1, W2 and b2
            W1 and b1 represent the weights and bias for the hidden layer of the network
            W2 and b2 represent the weights and bias for the output layer of the network

    Returns:
        A 3 element tuple containing:
            1. A numpy array of the activations (after the sigmoid) of the hidden layer
            2. A numpy array The output (after the softmax) of the output layer
            3. The average loss for these data elements
    """

    W1 = params['W1']
    b1 = params['b1']
    W2 = params['W2']
    b2 = params['b2']

    data_size = data.shape[0]
    a_1 = sigmoid(data.dot(W1)+b1)
    y_hat = softmax(a_1.dot(W2)+b2)
    SUM_CE = np.sum(-labels*np.log(y_hat))
    cost = (1/data_size) * SUM_CE
    return a_1, y_hat, cost

def backward_prop(data, labels, params, forward_prop_func):
    """
    Implement the backward propegation gradient computation step for a neural network

    Args:
        data: A numpy array containing the input
        labels: A 2d numpy array containing the labels
        params: A dictionary mapping parameter names to numpy arrays with the parameters.
            This numpy array will contain W1, b1, W2 and b2
            W1 and b1 represent the weights and bias for the hidden layer of the network
            W2 and b2 represent the weights and bias for the output layer of the network
        forward_prop_func: A function that follows the forward_prop API above

    Returns:
        A dictionary of strings to numpy arrays where each key represents the name of a weight
        and the values represent the gradient of the loss with respect to that weight.

        In particular, it should have 4 elements:
            W1, W2, b1, and b2
    """
    return backward_prop_regularized(data, labels, params, forward_prop_func, 0)

def backward_prop_regularized(data, labels, params, forward_prop_func, reg):
    """
    Implement the backward propagation gradient computation step for a neural network

    Args:
        data: A numpy array containing the input
        labels: A 2d numpy array containing the labels
        params: A dictionary mapping parameter names to numpy arrays with the parameters.
            This numpy array will contain W1, b1, W2 and b2
            W1 and b1 represent the weights and bias for the hidden layer of the network
            W2 and b2 represent the weights and bias for the output layer of the network
        forward_prop_func: A function that follows the forward_prop API above
        reg: The regularization strength (lambda)

    Returns:
        A dictionary of strings to numpy arrays where each key represents the name of a weight
        and the values represent the gradient of the loss with respect to that weight.

        In particular, it should have 4 elements:
            W1, W2, b1, and b2
    """
    W1 = params['W1']
    b1 = params['b1']
    W2 = params['W2']
    b2 = params['b2']

    data_size = data.shape[0]
    a_1, y_hat, cost = forward_prop_func(data, labels, params)
    gradient_W2 = 1 / data_size * a_1.T.dot(y_hat - labels) + 2 * reg * W2
    gradient_b2 = 1 / data_size * np.sum(y_hat - labels, axis=0)
    gradient_W1 = 1 / data_size * data.T.dot((y_hat - labels).dot(W2.T) * a_1 * (1 - a_1)) + 2 * reg * W1
    gradient_b1 = 1 / data_size * np.sum((y_hat - labels).dot(W2.T) * a_1 * (1 - a_1), axis=0)
    gradient = {}
    gradient['W1'] = gradient_W1
    gradient['W2'] = gradient_W2
    gradient['b1'] = gradient_b1
    gradient['b2'] = gradient_b2

    return gradient

def gradient_descent_epoch(train_data, train_labels, learning_rate, batch_size, params, forward_prop_func, backward_prop_func):
    """
    Perform one epoch of gradient descent on the given training data using the provided learning rate.

    This code should update the parameters stored in params.
    It should not return anything

    Args:
        train_data: A numpy array containing the training data
        train_labels: A numpy array containing the training labels
        learning_rate: The learning rate
        batch_size: The amount of items to process in each batch
        params: A dict of parameter names to parameter values that should be updated.
        forward_prop_func: A function that follows the forward_prop API
        backward_prop_func: A function that follows the backwards_prop API

    Returns: This function returns nothing.
    """

    (n_examples, d_value) = train_data.shape
    print('n_examples: ', n_examples)
    print('batch_size: ', batch_size)
    for index in range(n_examples // batch_size):
        data_pointer = index * batch_size
        gradient = backward_prop_func(
            train_data[data_pointer:data_pointer + batch_size, :],
            train_labels[data_pointer:data_pointer + batch_size, :],
            params, forward_prop_func)

        params['W1'] = params['W1'] - learning_rate * gradient['W1']
        params['W2'] = params['W2'] - learning_rate * gradient['W2']
        params['b1'] = params['b1'] - learning_rate * gradient['b1']
        params['b2'] = params['b2'] - learning_rate * gradient['b2']

    # This function does not return anything
    return

def nn_train(
    train_data, train_labels, dev_data, dev_labels,
    get_initial_params_func, forward_prop_func, backward_prop_func,
    num_hidden=300, learning_rate=0.001, num_epochs=30, batch_size=12):
    """
    Train the fully connected neural network
    Args:
        train_data: Training data examples
        train_labels: Training labels examples
        dev_data: Dev data examples
        dev_labels: DEv labels examples
        get_initial_params_func: Function to initialize parameters
        forward_prop_func: forward propagation function of NN
        backward_prop_func: backward propagation function of NN
        num_hidden: number of hidden units in hidden layer
        learning_rate: learning rate of algorithm
        num_epochs: number of epochs to be trained
        batch_size: batch size (not all samples at the same iteration)

    Returns:
        params, cost_train, cost_dev, accuracy_train, accuracy_dev
    """

    (nexp, dim) = train_data.shape

    params = get_initial_params_func(dim, num_hidden, 7)

    cost_train = []
    cost_dev = []
    accuracy_train = []
    accuracy_dev = []
    for epoch in range(num_epochs):
        print('epoch: ',epoch)
        gradient_descent_epoch(train_data, train_labels,
            learning_rate, batch_size, params, forward_prop_func, backward_prop_func)

        h, output, cost = forward_prop_func(train_data, train_labels, params)
        cost_train.append(cost)
        accuracy_train.append(compute_accuracy(output,train_labels))
        h, output, cost = forward_prop_func(dev_data, dev_labels, params)
        cost_dev.append(cost)
        accuracy_dev.append(compute_accuracy(output, dev_labels))

    return params, cost_train, cost_dev, accuracy_train, accuracy_dev

def nn_test(data, labels, params, name):
    """
    Used to test the trained NN
    Args:
        data: imput data
        labels: input labels
        params: trained parameters

    Returns:
        accuracy of the trained NN
    """
    h, output, cost = forward_prop(data, labels, params)
    pred_y = np.argmax(output, axis=1)
    test_y = np.argmax(labels, axis=1)
    print(pred_y)
    print(test_y)
    accuracy = compute_accuracy(output, labels)
    precision = precision_score(test_y, pred_y, zero_division=1, average='weighted')
    recall = recall_score(test_y, pred_y, average='weighted')
    f1 = f1_score(test_y, pred_y, zero_division=1, average='weighted', labels=np.unique(output))
    cm = confusion_matrix(test_y, pred_y)
    print('Confusion Matrix', cm)
    cmd = ConfusionMatrixDisplay(confusion_matrix=cm)
    cmd.plot()
    plt.savefig('NN_Confusion', dpi=300)
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
    plt.title('NN curve')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive rate')
    plt.legend(loc='best')
    plt.savefig(str(name) + '_NN', dpi=300)

    return accuracy, precision, recall, f1, cm

def compute_accuracy(output, labels):
    """
    Computes the accuracy of the NN
    Args:
        output: predicted values
        labels: real values

    Returns:
        accuracy of the trained NN
    """
    accuracy = (np.argmax(output,axis=1) ==
        np.argmax(labels,axis=1)).sum() * 1. / labels.shape[0]
    return accuracy

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

def get_labels():
    """
    Read labels from CSV (C-Level real values).
    Returns:
        A vector of labels.
    """
    train_labels = np.loadtxt('labels.csv', delimiter=',')
    return train_labels

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
            train_data = np.asarray(img_file.getdata(), dtype=np.int).reshape(750000,1)
        else:
            temp_array = np.asarray(img_file.getdata(), dtype=np.int).reshape(750000,1)
            train_data = np.append(train_data,temp_array,axis=1)
        print(train_data.T.shape)
    return train_data.T

def run_train_test(name, all_data, all_labels, backward_prop_func, num_epochs, plot=True):
    """
    Run the train test after epochs (with and without reglarization.
    Args:
        name: name of train test.
        all_data: dictionary containing all data (training, val and test sets).
        all_labels: dictionary containing all labels (training, val and test sets).
        backward_prop_func: function that implemented backward propagation.
        num_epochs: number of epochs to run.
        plot: boolean indication to plot or not.

    Returns:
        Accuracy of the model.
    """
    params, cost_train, cost_dev, accuracy_train, accuracy_dev = nn_train(
        all_data['train'], all_labels['train'],
        all_data['dev'], all_labels['dev'],
        get_initial_params, forward_prop, backward_prop_func,
        num_hidden=300, learning_rate=0.001, num_epochs=num_epochs, batch_size=12
    )

    t = np.arange(num_epochs)

    if plot:
        fig, (ax1, ax2) = plt.subplots(2, 1)

        ax1.plot(t, cost_train,'r', label='train')
        ax1.plot(t, cost_dev, 'b', label='dev')
        ax1.set_xlabel('epochs')
        ax1.set_ylabel('loss')
        if name == 'baseline':
            ax1.set_title('Without Regularization')
        else:
            ax1.set_title('With Regularization')
        ax1.legend()

        ax2.plot(t, accuracy_train,'r', label='train')
        ax2.plot(t, accuracy_dev, 'b', label='dev')
        ax2.set_xlabel('epochs')
        ax2.set_ylabel('accuracy')
        ax2.legend()

        fig.savefig('./' + name + '.pdf')
        fig.clf()

    accuracy, precision, recall, f1, confusion = nn_test(all_data['test'], all_labels['test'], params, name)
    print('For model %s, got accuracy: %f' % (name, accuracy))
    print('For model %s, got precision: %f' % (name, precision))
    print('For model %s, got recall: %f' % (name, recall))
    print('For model %s, got f1: %f' % (name, f1))
    print('For model %s, got confusion: ' % (name))
    print(confusion)

    return accuracy

def classification(y_real, y_pred):
    """ Transforms y_pred on buckets based on y_real values.

    Returns:
        Numpy array of classified labels (n_examples,).
    """
    edge_val = np.histogram_bin_edges(y_real, bins=7, range=None, weights=None)
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

# def test():
#     data = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10]).reshape(5, 2)
#     w = np.array([1, 2, 3, 4, 5, 6]).reshape(2, 3)
#     b = np.array([1, 1, 2]).reshape(3, )
#     dot_product = data.dot(w) + b
#     dot_product_2 = (w.T.dot(data.T)).T + b
#     return

def main(plot=True):
    test = np.array([[ 0,  0,  6,  0,  1,  0,  0],
     [ 1,  0, 11,  0,  3,  0,  0],
     [ 0,  0,  3,  0,  0,  0,  0],
     [ 0,  0, 13,  0,  0,  0,  0],
     [ 0,  0, 10,  0,  3,  0,  0],
     [ 0,  0,  1,  0,  0,  0,  0],
     [ 0,  0,  3,  0,  0,  0,  0]])
    cmd = ConfusionMatrixDisplay(confusion_matrix=test)
    font = {'family': 'Arial',
            'weight': 'bold',
            'size': 20}
    plt.rc('font', **font)
    cmd.plot()
    plt.savefig('NN_Confusion', dpi=300)
    quit()
    parser = argparse.ArgumentParser(description='Train a nn model.')
    parser.add_argument('--num_epochs', type=int, default=30)

    args = parser.parse_args()

    np.random.seed(100)
    orig_x, orig_y = read_data()
    orig_x = orig_x / 255.
    orig_y = classification(orig_y, orig_y)
    orig_y = one_hot_labels(orig_y)
    # Create the training sample
    train_x, val_test_x, train_y, val_test_y = train_test_split(orig_x, orig_y, test_size=0.2, random_state=1)
    # Split the remaining observations into validation and test
    val_x, test_x, val_y, test_y = train_test_split(val_test_x, val_test_y, test_size=0.5, random_state=1)

    p = np.random.permutation(train_x.shape[0])
    train_x = train_x[p,:]
    train_y = train_y[p,:]

    mean = np.mean(orig_x)
    std = np.std(orig_x)
    train_x = (train_x - mean) / std
    val_x = (val_x - mean) / std

    all_data = {
        'train': train_x,
        'dev': val_x,
        'test': test_x
    }

    all_labels = {
        'train': train_y,
        'dev': val_y,
        'test': test_y,
    }

    baseline_acc = run_train_test('baseline', all_data, all_labels, backward_prop, args.num_epochs, plot)
    reg_acc = run_train_test('regularized', all_data, all_labels,
        lambda a, b, c, d: backward_prop_regularized(a, b, c, d, reg=0.0001),
        args.num_epochs, plot)

    return baseline_acc, reg_acc

if __name__ == '__main__':
    main()
