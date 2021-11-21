import numpy as np
import math
import util
import scipy.stats as stats
import pylab

np.seterr(all='raise')

factor = 2.0

class LinearModel(object):
    """Base class for linear models."""

    def __init__(self, theta=None):
        """
        Args:
            theta: Weights vector for the model.
        """
        self.theta = theta

    def fit(self, X, y):
        self.theta = np.linalg.solve(np.dot(X.T, X), np.dot(X.T, y))

    def create_poly(self, k, x):
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
        return np.dot(X, self.theta)

def ceil(x):
    return math.ceil(x/0.7)

def run_exp(train_path, valid_path):
    # *** START CODE HERE ***
    lm = LinearModel()
    train_x, train_y = util.load_dataset(train_path, add_intercept=False)
    test_x, test_y = util.load_dataset(valid_path, add_intercept=False)

    new_train_x = lm.create_poly(5, train_x)
    lm.fit(new_train_x, train_y)
    new_test_x = lm.create_poly(5, test_x)
    predicted_y = lm.predict(new_test_x)
    rmsd = math.sqrt(np.square(test_y-predicted_y).sum()/455)
    print(rmsd)
    ceil_v = np.vectorize(ceil)
    accuracy = (ceil_v(test_y) == ceil_v(predicted_y)).sum() / predicted_y.shape[0]
    print(accuracy)
    # *** END CODE HERE ***

def main(train_path, valid_path):
    run_exp(train_path, valid_path)

if __name__ == '__main__':
    main(train_path='train_x.csv',
         valid_path='test_x.csv')
