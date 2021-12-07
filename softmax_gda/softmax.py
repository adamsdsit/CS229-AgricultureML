import numpy as np
import util
import scipy.sparse
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, confusion_matrix
from sklearn.metrics import roc_curve
from sklearn.preprocessing import StandardScaler
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import pandas as pd
import tensorflow as tf
import os
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, confusion_matrix
from sklearn.metrics import roc_curve
from sklearn.metrics import ConfusionMatrixDisplay
from sklearn import preprocessing

def main(image_path):
    """Problem: Multiclass Softmax

    Args:
        image_path: Path to CSV file containing dataset for training.
    """
    # test = np.array([[0,4,0,9,1,0,0],
    #                  [0,9,0,3,1,0,0],
    #                  [1,7,0,2,2,0,0],
    #                  [2,12,0,5,0,0,0],
    #                  [0,10,0,8,3,0,0],
    #                  [0,1,0,0,0,0,0],
    #                  [0,2,0,0,0,0,0]])
    # cmd = ConfusionMatrixDisplay(confusion_matrix=test)
    # font = {'family': 'Arial',
    #         'weight': 'bold',
    #         'size': 20}
    # plt.rc('font', **font)
    # cmd.plot()
    # plt.savefig('Softmax_Confusion', dpi=300)
    # quit()
    # Read the file of variables
    orig_x, orig_y = util.load_dataset(image_path, add_intercept=False)
    orig_y = classification(orig_y,orig_y)
    # Create the training sample
    # WE're using cross-validation instead of splitting training and validation sets
    train_x, val_test_x, train_y, val_test_y = train_test_split(orig_x, orig_y, test_size=0.3, random_state=1)
    # Split the remaining observations into validation and test
    val_x, test_x, val_y, test_y = train_test_split(val_test_x, val_test_y, test_size=0.5, random_state=1)

    scaler = StandardScaler()
    # train_x = scaler.fit_transform(train_x)
    # val_x = scaler.fit_transform(val_x)
    # test_x = scaler.fit_transform(test_x)

    n_examples, n = train_x.shape
    # Train a softmax regression classifier
    clf = SoftmaxRegression()
    clf.fit(train_x, train_y)
    prob, pred_y = clf.predict(test_x)
    print(prob.shape)

    # Calculate accuracy, precision, recall and confusion matrix
    print('Training Accuracy: ', clf.getAccuracy(train_x, train_y))
    print('Test Accuracy: ', accuracy_score(test_y, pred_y))
    print('Test Precision: ', precision_score(test_y, pred_y, zero_division=0, average='weighted'))
    print('Test Recall: ', recall_score(test_y, pred_y, average='weighted'))
    print('Test f1', f1_score(test_y, pred_y, zero_division=1, average='weighted', labels=np.unique(pred_y)))
    cm = confusion_matrix(test_y, pred_y)
    print('Confustion Matrix', cm)
    cmd = ConfusionMatrixDisplay(confusion_matrix=cm)
    cmd.plot()
    plt.savefig('Softmax_Confusion', dpi=300)
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
    plt.title('Softmax ROC curve')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive rate')
    plt.legend(loc='best')
    plt.savefig('Softmax ROC', dpi=300)

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
        """Run solver to fit softmax model.

        Args:
            X: Training example inputs. Shape (n_examples, dim).
            y: Training example labels. Shape (n_examples,).
        """
        eps = 1e-5
        n_examples, n = x.shape
        self.learningRate = 1e-5
        losses = []
        if self.theta is None:
            self.theta = np.zeros([n,7])
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
        """Get loss value for each step of the iteration

        Args:
            x: Training example inputs.
            y: Training example labels.
        """
        n = x.shape[0]
        y_mat = self.oneHotIt(y)
        z = np.dot(x,self.theta)
        prob = self.softmax(z)
        loss = (-1./n) * np.sum(y_mat * np.log(prob))
        grad = (-1./n) * np.dot(x.T,(y_mat-prob))
        return loss, grad

    def oneHotIt(self, Y):
        """
        Convert y labels to one-hot representation.
        Args:
            Y: Training example labels.

        Returns:
            Labels in one hor representation
        """
        m = Y.shape[0]
        OHX = scipy.sparse.csr_matrix((np.ones(m), (Y.astype(int), np.array(range(m)))))
        OHX = np.array(OHX.todense()).T
        return OHX

    def softmax(self, z_examples):
        """
        Calculates the probability of each class.
        Args:
            z_examples: z-values for each sample in each class

        Returns:
            Array of probabilites of each class for each sample
        """
        z_examples = z_examples - np.max(z_examples)
        # Calculate the softmax value for each value, considering the sum only on second dimension
        softmax = (np.exp(z_examples).T / np.sum(np.exp(z_examples),axis=1)).T
        return softmax

    def predict(self, x):
        """
            Return the predictions of each sample from training example inputs.
        Args:
            x: Training example inputs.

        Returns:
            Array of probabilites for each class and predictions (max probability)
        """
        probs = self.softmax(np.dot(x, self.theta))
        preds = np.argmax(probs, axis=1)
        return probs, preds

    def getAccuracy(self, x, y):
        """
        Returns accuracy of the model,
        Args:
            x: Training example inputs.
            y: Training example labels.

        Returns:
            Accuracy of the model.
        """
        prob, prede = self.predict(x)
        accuracy = sum(prede == y) / (float(len(y)))
        return accuracy

if __name__ == '__main__':
    main(image_path='images.csv')