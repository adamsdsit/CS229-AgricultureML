import numpy as np
import matplotlib.pyplot as plt
import argparse
import os
from PIL import Image

def softmax(x):
    """
    Compute softmax function for a batch of input values.
    The first dimension of the input corresponds to the batch size. The second dimension
    corresponds to every class in the output. When implementing softmax, you should be careful
    to only sum over the second dimension.

    Important Note: You must be careful to avoid overflow for this function. Functions
    like softmax have a tendency to overflow when very large numbers like e^10000 are computed.
    You will know that your function is overflow resistent when it can handle input like:
    np.array([[10000, 10010, 10]]) without issues.

    Args:
        x: A 2d numpy float array of shape batch_size x number_of_classes

    Returns:
        A 2d numpy float array containing the softmax results of shape batch_size x number_of_classes
    """
    # *** START CODE HERE ***
    # Calculate the max value for each line of batch size
    max_value = np.max(x, axis=1)[:, np.newaxis]
    # Divide each value by the max of collection: EXP(x)/EXP(max(x)) = EXP(x-max(x))
    x = x - max_value
    # Calculate the EPX(X-max(X))
    exp_x = np.exp(x)
    # Calculate the softmax value for each value, considering the sum only on second dimension
    softmax = exp_x / np.sum(exp_x, axis=1)[:, np.newaxis]
    return softmax
    # *** END CODE HERE ***

def sigmoid(x):
    """
    Compute the sigmoid function for the input here.

    Args:
        x: A numpy float array

    Returns:
        A numpy float array containing the sigmoid results
    """
    # *** START CODE HERE ***
    sigmoid = 1 / (1 + np.exp(-x))
    return sigmoid
    # *** END CODE HERE ***

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

    As specified in the PDF, weight matrices should be initialized with a random normal distribution
    centered on zero and with scale 1.
    Bias vectors should be initialized with zero.

    Args:
        input_size: The size of the input data
        num_hidden: The number of hidden states
        num_output: The number of output classes

    Returns:
        A dict mapping parameter names to numpy arrays
    """

    # *** START CODE HERE ***
    return_dict = {}
    # W1 size is d x h: input_size x num_hidden; and it is a random normal distribution
    return_dict['W1'] = np.random.normal(size=(input_size, num_hidden))
    # b1 size is h x 1: num_hidden x 1
    return_dict['b1'] = np.zeros(num_hidden)
    # W2 size is h x 10: num_hidden x num_output(10)
    return_dict['W2'] = np.random.normal(size=(num_hidden, num_output))
    # b2 size is 10 x 1: num_output(10) x 1
    return_dict['b2'] = np.zeros(num_output)
    return return_dict
    # *** END CODE HERE ***

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
    # *** START CODE HERE ***

    #get W1, b1, W2 and b2 from params
    W1 = params['W1']
    b1 = params['b1']
    W2 = params['W2']
    b2 = params['b2']

    # It is important to note that data is batch_size x d (1000 x 784)
    data_size = data.shape[0]
    # The calculus of Z_1 and a_1
    # From question statement, W1 is d x h, d x hidden_size(units), (784 x 300)
    # In question statement, the formula for Z_1 is W1.T.dot(X) + b; observe that data = X.T
    # So in code we calculate data.dot(W1); This is important for broadcasting to work
    # data.dot(W1) is input_size x hidden_size, (1000 x 300)
    # Broadcasting makes b1 broadcast to all lines (numpy); we just add b1 (300,) to data.dot(W1).
    # The we get the activation value just applying sigmoid function
    # So activation (a_1) is input_size x hidden_size (units) = 1000 x 300
    a_1 = sigmoid(data.dot(W1)+b1)
    # The calculus of Z_2 and y_hat
    # From question statement, Z_2 is W2.T.dot(a_1)
    # Observe that W2 is num_hidden x num_output (300 x 10)
    # For the same reason (broadcasting), we calculate Z_2 as a_1.dot(W2) and then add b2(10,)
    # To calculate y_hat, we apply the softmax function
    # y_hat is input_size x num_output, (1000 x 10)
    y_hat = softmax(a_1.dot(W2)+b2)
    # Calculate the loss of mini-batch
    # data_size is 1000 (examples)
    # From question statement, cost is given by: 1/data_size * SUM(CE(y, y_hat)); y = labels
    SUM_CE = np.sum(-labels*np.log(y_hat))
    cost = (1/data_size) * SUM_CE
    return a_1, y_hat, cost
    # *** END CODE HERE ***

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
    # *** START CODE HERE ***
    return backward_prop_regularized(data, labels, params, forward_prop_func, 0)
    # *** END CODE HERE ***


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
    # *** START CODE HERE ***
    W1 = params['W1']
    b1 = params['b1']
    W2 = params['W2']
    b2 = params['b2']

    # It is important to note that data is batch_size x d (1000 x 784)
    data_size = data.shape[0]
    # We get the values from forward propagation
    a_1, y_hat, cost = forward_prop_func(data, labels, params)
    # Then we calculate the gradients w.r.t to W2, b2, W1, b1
    # Since it is a mini batch calculus, we need to take mean value among data_size
    # From lecture notes, gradient_W2 = 1 / data_size * (y_hat-y).dot(a_1.T)
    # Recall that a_1 is input_size x hidden_size (units) = 1000 x 300
    # y_hat, labels is input_size x num_output (units) = 1000 x 10
    # W2 and gradW2 is hidden_size x num_output = 300 x 10
    # So, in numpy, we have gradient_W2 = 1 / data_size * a_1.T.dot(y_hat-y)
    # We add the term 2 * lambda * W2 (partial derivative of regularization term w.r.t W2)
    gradient_W2 = 1 / data_size * a_1.T.dot(y_hat - labels) + 2 * reg * W2
    # From lecture notes, gradient_b2 = 1 / data_size * (y_hat-y)
    # Recall that y_hat, labels is input_size x num_output, (1000 x 10)
    # So we calculate the average gradient_b2_i through data_size samples (1000) for all num_output (i=1...10)
    # Then our gradient_b2 is num_output, (10,)
    gradient_b2 = 1 / data_size * np.sum(y_hat - labels, axis=0)
    # From lecture notes, gradient_W1 = (W2.T.dot(y_hat-y) * sigma').dot(X), with sigma' = sigma(1 - sigma)
    # W2 is hidden_size x num_output = 300 x 10
    # y_hat, labels is input_size x num_output (units) = 1000 x 10
    # Then (y_hat - y).dot(W2.T) * sigma' is input_size x hidden_size, (1000 x 300)
    # data is input_size x d, (1000 x 784)
    # data.T.dot((y_hat - labels).dot(W2.T) * sigma') is d x hidden_size, (784 x 300)
    # We add the term 2 * lambda * W1 (partial derivative of regularization term w.r.t W1)
    gradient_W1 = 1 / data_size * data.T.dot((y_hat - labels).dot(W2.T) * a_1 * (1 - a_1)) + 2 * reg * W1
    # From lecture notes, gradient_b1 = W2.T.dot(y_hat-y) * sigma', with sigma' = sigma(1 - sigma)
    # We must calculate the average gradient_b1_i for all num_output (i=1...300)
    gradient_b1 = 1 / data_size * np.sum((y_hat - labels).dot(W2.T) * a_1 * (1 - a_1), axis=0)
    gradient = {}
    gradient['W1'] = gradient_W1
    gradient['W2'] = gradient_W2
    gradient['b1'] = gradient_b1
    gradient['b2'] = gradient_b2

    return gradient
    # *** END CODE HERE ***

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

    # *** START CODE HERE ***
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
    # *** END CODE HERE ***

    # This function does not return anything
    return

def nn_train(
    train_data, train_labels, dev_data, dev_labels,
    get_initial_params_func, forward_prop_func, backward_prop_func,
    num_hidden=300, learning_rate=0.0007, num_epochs=90, batch_size=12):

    (nexp, dim) = train_data.shape

    params = get_initial_params_func(dim, num_hidden, 10)

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

def nn_test(data, labels, params):
    h, output, cost = forward_prop(data, labels, params)
    accuracy = compute_accuracy(output, labels)
    return accuracy

def compute_accuracy(output, labels):
    accuracy = (np.argmax(output,axis=1) ==
        np.argmax(labels,axis=1)).sum() * 1. / labels.shape[0]
    return accuracy

def one_hot_labels(labels):
    one_hot_labels = np.zeros((labels.size, 10))
    one_hot_labels[np.arange(labels.size),labels.astype(int)] = 1
    return one_hot_labels

def read_data(images_file, labels_file):
    x = np.loadtxt(images_file, delimiter=',')
    y = np.loadtxt(labels_file, delimiter=',')
    return x, y

def read_data_2():
    x = get_train_data()
    y = get_train_labels()
    return x,y

def read_data_3():
    x = get_test_data()
    y = get_test_labels()
    return x,y

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

def run_train_test(name, all_data, all_labels, backward_prop_func, num_epochs, plot=True):
    params, cost_train, cost_dev, accuracy_train, accuracy_dev = nn_train(
        all_data['train'], all_labels['train'],
        all_data['dev'], all_labels['dev'],
        get_initial_params, forward_prop, backward_prop_func,
        num_hidden=300, learning_rate=0.0007, num_epochs=num_epochs, batch_size=12
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

    accuracy = nn_test(all_data['test'], all_labels['test'], params)
    print('For model %s, got accuracy: %f' % (name, accuracy))

    return accuracy

# def test():
#     data = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10]).reshape(5, 2)
#     w = np.array([1, 2, 3, 4, 5, 6]).reshape(2, 3)
#     b = np.array([1, 1, 2]).reshape(3, )
#     dot_product = data.dot(w) + b
#     dot_product_2 = (w.T.dot(data.T)).T + b
#     return

def main(plot=True):
    # test()
    parser = argparse.ArgumentParser(description='Train a nn model.')
    parser.add_argument('--num_epochs', type=int, default=90)

    args = parser.parse_args()

    np.random.seed(100)
    # train_data, train_labels = read_data('./images_train.csv', './labels_train.csv')
    train_data, train_labels = read_data_2()
    train_labels = one_hot_labels(train_labels)
    p = np.random.permutation(455)
    train_data = train_data[p,:]
    train_labels = train_labels[p,:]

    dev_data = train_data[0:91,:]
    dev_labels = train_labels[0:91,:]
    train_data = train_data[91:,:]
    train_labels = train_labels[91:,:]

    mean = np.mean(train_data)
    std = np.std(train_data)
    train_data = (train_data - mean) / std
    dev_data = (dev_data - mean) / std

    # test_data, test_labels = read_data('./images_test.csv', './labels_test.csv')
    test_data, test_labels = read_data_3()
    test_labels = one_hot_labels(test_labels)
    test_data = (test_data - mean) / std

    all_data = {
        'train': train_data,
        'dev': dev_data,
        'test': test_data
    }

    all_labels = {
        'train': train_labels,
        'dev': dev_labels,
        'test': test_labels,
    }

    baseline_acc = run_train_test('baseline', all_data, all_labels, backward_prop, args.num_epochs, plot)
    reg_acc = run_train_test('regularized', all_data, all_labels,
        lambda a, b, c, d: backward_prop_regularized(a, b, c, d, reg=0.0001),
        args.num_epochs, plot)

    return baseline_acc, reg_acc

if __name__ == '__main__':
    main()
