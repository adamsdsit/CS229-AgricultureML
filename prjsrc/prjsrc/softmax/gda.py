import numpy as np
import util
import math

def main(train_path, valid_path):

    # Load dataset
    x_train, y_train = util.load_dataset(train_path, add_intercept=False)
    # Train a GDA classifier
    clf = GDA()
    x_train_normalized = np.log((x_train+5))
    x_norm_2 = (x_train - x_train.min(0)) / x_train.ptp(0)
    clf.fit(x_train, y_train)
    # Plot decision boundary on validation set
    x_validation, y_validation = util.load_dataset(valid_path, add_intercept=True)
    x_validation_normalized = np.log((x_validation+5))
    # x_valid_norm2 = (x_validation - x_validation.min(0)) / x_validation.ptp(0)
    y_predicted = clf.predict_2(x_validation)
    accuracy = (y_validation == y_predicted).sum() / len(y_predicted)
    print(accuracy)
    print(y_predicted)
    print(y_validation)
    # print(clf.score(x_validation, y_validation))
    # util.plot(x_validation, y_validation, clf.theta, plot_path)

class GDA:
    """Gaussian Discriminant Analysis.

    Example usage:
        > clf = GDA()
        > clf.fit(x_train, y_train)
        > clf.predict(x_eval)
    """
    def __init__(self, eps=1e-10, verbose=True):
        self.eps = eps
        self.verbose = verbose

    def fit(self, x, y):
        # y_classes is a (9,) vector; y_counts too;
        self.y_classes, y_counts = np.unique(y, return_counts=True)
        # Find phi - probability of each class
        self.phi = 1.0 * y_counts/len(y)
        # Find mu_k - dimension (9,22)
        self.mu = np.array([x[y == k].mean(axis=0) for k in self.y_classes])
        # Find sigma - dimension (22,22)
        self.sigma = self.compute_sigma(x, y)
        self.sigma_inv = np.linalg.inv(self.sigma)
        self.theta = np.zeros([int(x.shape[1] + 1), self.y_classes.shape[0]])
        theta_0 = (- 0.5 * ((self.mu).dot(self.sigma_inv).dot(self.mu.T)) + np.log(self.phi)).sum(axis=1)
        self.theta[0] = theta_0
        self.theta[1:] = self.sigma_inv.dot(self.mu.T)

    def compute_sigma(self, x, y):
        x_minus_mu = x.copy().astype('float64')
        for i in range(len(self.mu)):
            x_minus_mu[y==self.y_classes[i]] -= self.mu[i]
        return (1 / len(y)) * x_minus_mu.T.dot(x_minus_mu)

    def predict(self, X):
        return np.apply_along_axis(self.get_prob, 1, X)

    def score(self, X, y):
        return (self.predict(X) == y).mean()

    def predict_2(self, x):
        new_x = x.dot(self.theta)
        max_value = np.max(new_x, axis=1)[:, np.newaxis]
        new_x = new_x - max_value
        new_x = np.exp(new_x)
        #sums = np.sum(new_x, axis=1)[:, np.newaxis]
        #p = new_x / sums
        return new_x.argmax(axis=1)

    def get_prob(self, x):
        new_p = np.exp(-0.5 * np.sum((x - self.mu).dot(self.sigma_inv) * (x - self.mu), axis=1)) * self.phi
        return np.argmax(new_p)

if __name__ == '__main__':
    main(train_path='train_x_normalized.csv',
         valid_path='test_x_normalized.csv')

