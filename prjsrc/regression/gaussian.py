import numpy as np
import random
import util
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from sklearn import preprocessing
from sklearn.preprocessing import StandardScaler

class GaussianRegression:
    """Base class for GLM Gaussian ."""
    def __init__(self, lambda_coefficient=0.02, step_size=1e-3, max_iter=300000, eps=1e-5,
                 theta_0=None, verbose=True):
        """
        Args:
            theta: Weights vector for the model.
            step_size: learning rate of the model
            max_iter: maximum number of iterations
            eps: precision of the algorithm
            verbose: present or not more debug information
        """
        self.theta = theta_0
        self.step_size = step_size
        self.max_iter = max_iter
        self.eps = eps
        self.verbose = verbose
        self.lambda_coefficient = lambda_coefficient

    def fit(self, x, y):
        """Run solver to fit GLM Gaussian (same as linear regression).

            Args:
                X: Training example inputs. Shape (n_examples, dim).
                y: Training example labels. Shape (n_examples,).
        """
        n_examples, n = x.shape
        if self.theta is None:
            self.theta = np.zeros(n)
        num_iter = 0
        while True:
            num_iter += 1
            theta = np.copy(self.theta)
            eta = x.dot(theta)
            gradient = (y - eta).dot(x) - self.lambda_coefficient*self.theta
            self.theta = self.theta + self.step_size * gradient
            # print(num_iter)
            # print(np.linalg.norm((self.theta - theta), ord=2))
            if num_iter >= self.max_iter or np.linalg.norm((self.theta - theta), ord=2) < self.eps:
                break

    def predict(self, x):
        """
            Make a prediction given new inputs x.
            Returns the numpy array of the predictions.

            Args:
                X: Inputs of shape (n_examples, dim).

            Returns:
                Outputs of shape (n_examples,).
        """
        y_predicted = x.dot(self.theta)
        return y_predicted

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

def main(image_path):
    # Establish a random seed
    random.seed(0)
    # Read the file of variables
    orig_x, orig_y = util.load_dataset(image_path, add_intercept=False)
    # Create the training sample
    # WE're using cross-validation instead of splitting training and validation sets
    train_x, val_test_x, train_y, val_test_y = train_test_split(orig_x, orig_y, test_size=0.3, random_state=1)
    # Split the remaining observations into validation and test
    val_x, test_x, val_y, test_y = train_test_split(val_test_x, val_test_y, test_size=0.5, random_state=1)

    lr = GaussianRegression()
    train_x = preprocessing.normalize(train_x)
    val_x = preprocessing.normalize(val_x)
    test_x = preprocessing.normalize(test_x)
    scaler = StandardScaler()
    # train_x = scaler.fit_transform(train_x)
    # val_x = scaler.fit_transform(val_x)
    # test_x = scaler.fit_transform(test_x)
    lr.fit(train_x, train_y)

    # Make predictions on the training, validation and test sets
    train_y_pred = lr.predict(train_x)
    val_y_pred = lr.predict(val_x)
    test_y_pred = lr.predict(test_x)

    # Print empirical risk on both sets
    print('MSE on training set:')
    print(mean_squared_error(train_y, train_y_pred))
    print('MSE on validation set:')
    MSE = mean_squared_error(val_y, val_y_pred)
    print(MSE)
    print('MSE on test set:')
    print(mean_squared_error(test_y, test_y_pred))
    print('')

    # Print R squared on both sets
    print('R squared on training set:')
    print(r2_score(train_y, train_y_pred))
    print('R squared on validation set:')
    print(r2_score(val_y, val_y_pred))
    print('R squared on test set:')
    print(r2_score(test_y, test_y_pred))

    # Measure accuracy for classification problem
    print('Accuracy: ')
    accuracy = (classification(orig_y, test_y) == classification(orig_y, test_y_pred)).sum() / test_y.shape[0]
    print(accuracy)
    quit()

    # Randomly choose 10 inputs without replacement
    input_indices = random.sample(range(0, train_x.shape[1]), 22)
    # Train the model using the training set
    lr.fit(train_x[:, input_indices], train_y)

    # For 1000 times, randomly choose 22 inputs, estimate regression,
    # and save if performance on validation is better than that of
    # previous regressions
    betterResult = False
    for j in range(0, 10):
        print(j)
        lr_j = GaussianRegression()
        input_indices_j = random.sample(range(0, train_x.shape[1]), 22)
        lr_j.fit(train_x[:, input_indices_j], train_y)
        val_y_pred_j = lr_j.predict(val_x[:, input_indices_j])
        MSE_j = mean_squared_error(val_y, val_y_pred_j)
        if MSE_j < MSE:
            betterResult = True
            input_indices = input_indices_j
            lr = lr_j
            MSE = MSE_j

    if (betterResult):
        # Make predictions on the train, validation and test sets using better results
        train_y_pred = lr.predict(train_x[:, input_indices])
        val_y_pred = lr.predict(val_x[:, input_indices])
        test_y_pred = lr.predict(test_x[:, input_indices])

        # Print empirical risk on all sets
        print('MSE on training set:')
        print(mean_squared_error(train_y, train_y_pred))
        print('MSE on validation set:')
        print(mean_squared_error(val_y, val_y_pred))
        print('MSE on test set:')
        print(mean_squared_error(test_y, test_y_pred))
        print('')

        # Print R squared on all sets
        print('R squared on training set:')
        print(r2_score(train_y, train_y_pred))
        print('R squared on validation set:')
        print(r2_score(val_y, val_y_pred))
        print('R squared on test set:')
        print(r2_score(test_y, test_y_pred))

    # Measure accuracy for classification problem
    print('Accuracy: ')
    accuracy = (classification(orig_y, test_y) == classification(orig_y, test_y_pred)).sum() / test_y.shape[0]
    print(accuracy)

if __name__ == '__main__':
    main(image_path='images.csv')