import numpy as np
import util
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
import random
from sklearn import preprocessing

class FeaturesMapModel(object):
    """Base class for linear models."""

    def __init__(self, theta=None):
        """
        Args:
            theta: Weights vector for the model.
        """
        self.theta = theta

    def fit(self, X, y):
        """Run solver to fit linear model with features maps. You have to update the value of
        self.theta using the normal equations.

        Args:
            X: Training example inputs. Shape (n_examples, dim).
            y: Training example labels. Shape (n_examples,).
        """
        inverse = np.linalg.pinv(np.dot(X.T, X))
        self.theta = inverse.dot(np.dot(X.T, y))
        # self.theta = np.linalg.solve(np.dot(X.T, X), np.dot(X.T, y))

    def create_poly(self, k, x):
        """ Create features for each variable: x + x^2 + log(x).

        Returns:
            List of features based on variables.
        """
        num_columns = x.shape[1]
        new_x = np.array([])
        for index_column in range(num_columns):
            x_values = x[:, index_column]
            for i in range(1, k + 1):
                pow_x = np.power(x_values, i)
                pow_x = pow_x[:, np.newaxis]
                if new_x.size == 0:
                    new_x = pow_x
                else:
                    new_x = np.column_stack((new_x,pow_x))
            extra = np.log(x_values)
            new_x = np.c_[new_x, extra]
        return new_x

    def predict(self, X):
        """
        Make a prediction given new inputs x.
        Returns the numpy array of the predictions.

        Args:
            X: Inputs of shape (n_examples, dim).

        Returns:
            Outputs of shape (n_examples,).
        """
        return np.dot(X, self.theta)

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

def run_exp(image_path):
    random.seed(0)
    orig_x, orig_y = util.load_dataset(image_path, add_intercept=True)
    # Create the training sample
    train_x, val_test_x, train_y, val_test_y = train_test_split(orig_x, orig_y, test_size=0.3, random_state=1)
    # Split the remaining observations into validation and test
    val_x, test_x, val_y, test_y = train_test_split(val_test_x, val_test_y, test_size=0.5, random_state=1)

    # Create the linear model and predict the values
    lr = FeaturesMapModel()
    train_x = lr.create_poly(2, train_x)
    val_x = lr.create_poly(2, val_x)
    test_x = lr.create_poly(2, test_x)
    # train_x = preprocessing.normalize(train_x)
    # val_x = preprocessing.normalize(val_x)
    # test_x = preprocessing.normalize(test_x)

    # Make predictions on the training and validation sets
    lr.fit(train_x, train_y)
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
    print('')

    # Randomly choose 10 inputs without replacement
    input_indices = random.sample(range(0, train_x.shape[1]), 7)
    # Train the model using the training set
    lr.fit(train_x[:, input_indices], train_y)

    # For 1000 times, randomly choose 7 inputs, estimate regression,
    # and save if performance on validation is better than that of
    # previous regressions;
    betterResult = False

    for j in range(0, 1000):
        lr_j = FeaturesMapModel()
        input_indices_j = random.sample(range(0, train_x.shape[1]), 7)
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

    # Export results
    # np.savetxt("predicted_y.csv", predicted_y, delimiter=",")
    # *** END CODE HERE ***

def main(image_path):
    run_exp(image_path)

if __name__ == '__main__':
    main(image_path='images.csv')