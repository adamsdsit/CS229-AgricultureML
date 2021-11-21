import numpy as np
import util
import scipy.sparse
import matplotlib.pyplot as plt

def main(train_path, valid_path):
    """Problem: Logistic regression with Newton's Method.

    Args:
        train_path: Path to CSV file containing dataset for training.
        valid_path: Path to CSV file containing dataset for validation.
    """
    x_train, y_train = util.load_dataset(train_path, add_intercept=True)
    n_examples, n = x_train.shape
    # Train a softmax regression classifier
    clf = SoftmaxRegression()
    clf.fit(x_train, y_train)

    # Plot decision boundary on top of validation set set
    x_test, y_test = util.load_dataset(valid_path, add_intercept=True)
    print('Training Accuracy: ', clf.getAccuracy(x_train, y_train))
    print('Test Accuracy: ', clf.getAccuracy(x_test, y_test))
    classWeightsToVisualize = 3
    plt.show(clf.theta)

class SoftmaxRegression:
    """Softmax regression with Newton's Method as the solver.

    Example usage:
        > clf = Softmax()
        > clf.fit(x_train, y_train)
        > clf.predict(x_eval)
    """
    def __init__(self, learning_rate=0.01, max_iter=10000, eps=1e-5,
                 theta_0=None, verbose=True):
        """
        Args:
            learning_rate: Step size for iterative solvers only.
            max_iter: Maximum number of iterations for the solver.
            eps: Threshold for determining convergence.
            theta_0: Initial guess for theta. If None, use the zero vector.
            verbose: Print loss values during training.
        """
        self.theta = theta_0
        self.learningRate = learning_rate
        self.max_iter = max_iter
        self.eps = eps
        self.verbose = verbose

    def fit(self, x, y):
        eps = 1e-5
        n_examples, n = x.shape
        self.learningRate = 1e-5
        losses = []
        if self.theta is None:
            self.theta = np.zeros([n,10])
        iter = 0
        while True:
            prev_theta = np.copy(self.theta)
            loss, grad = self.getLoss(x,y)
            losses.append(loss)
            self.theta = self.theta - self.learningRate * grad
            if np.linalg.norm((self.theta - prev_theta), ord=1) < eps:
                break
            if iter > self.max_iter:
                break
            iter += 1
        print(iter)
        plt.plot(losses)
        plt.show()

    def getLoss(self, x, y):
        n = x.shape[0]
        y_mat = self.oneHotIt(y)
        z = np.dot(x,self.theta)
        prob = self.softmax(z)
        loss = (-1./n) * np.sum(y_mat * np.log(prob))
        grad = (-1./n) * np.dot(x.T,(y_mat-prob))
        return loss, grad

    def oneHotIt(self, Y):
        m = Y.shape[0]
        OHX = scipy.sparse.csr_matrix((np.ones(m), (Y.astype(int), np.array(range(m)))))
        OHX = np.array(OHX.todense()).T
        return OHX

    def softmax(self, z_examples):
        z_examples = z_examples - np.max(z_examples)
        # Calculate the softmax value for each value, considering the sum only on second dimension
        softmax = (np.exp(z_examples).T / np.sum(np.exp(z_examples),axis=1)).T
        return softmax

    def predict(self, x):
        probs = self.softmax(np.dot(x, self.theta))
        preds = np.argmax(probs, axis=1)
        return probs, preds

    def getAccuracy(self, x, y):
        prob, prede = self.predict(x)
        accuracy = sum(prede == y) / (float(len(y)))
        return accuracy

if __name__ == '__main__':
    main(train_path='train_x.csv',
         valid_path='test_x.csv')