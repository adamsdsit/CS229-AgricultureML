import numpy as np
import util
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, confusion_matrix
from sklearn.metrics import roc_curve
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import ConfusionMatrixDisplay
from sklearn import preprocessing

def main(image_path):
    """Problem: Multiclass GDA

        Args:
            image_path: Path to CSV file containing dataset for training.
        """
    # test = np.array([[ 5,  5,  1,  2,  1,  0,  0],
    #                  [ 3,  6,  2,  0,  2,  0,  0],
    #                  [ 2,  4,  3,  3,  0,  0,  0],
    #                  [ 4,  4,  3,  5,  3,  0,  0],
    #                  [ 2,  3,  1,  4, 11,  0,  0],
    #                  [ 0,  0,  0,  0,  1,  0,  0],
    #                  [ 0,  2,  0,  0,  0,  0,  0]])
    # cmd = ConfusionMatrixDisplay(confusion_matrix=test)
    # font = {'family': 'Arial',
    #         'weight': 'bold',
    #         'size': 20}
    # plt.rc('font', **font)
    # cmd.plot()
    # plt.savefig('GDA_Confusion', dpi=300)
    # quit()
    # Read the file of variables
    orig_x, orig_y = util.load_dataset(image_path, add_intercept=False)
    orig_y = classification(orig_y, orig_y)
    # Create the training sample
    # WE're using cross-validation instead of splitting training and validation sets
    train_x, val_test_x, train_y, val_test_y = train_test_split(orig_x, orig_y, test_size=0.3, random_state=1)
    # Split the remaining observations into validation and test
    val_x, test_x, val_y, test_y = train_test_split(val_test_x, val_test_y, test_size=0.5, random_state=1)

    scaler = StandardScaler()
    # train_x = scaler.fit_transform(train_x)
    # val_x = scaler.fit_transform(val_x)
    # test_x = scaler.fit_transform(test_x)

    # Train a GDA classifier
    clf = GDA()
    x_train_normalized = np.log((train_x+5))
    x_norm_2 = (train_x - train_x.min(0)) / train_x.ptp(0)
    clf.fit(train_x, train_y)
    # Plot decision boundary on validation set
    x_validation_normalized = np.log((val_x+5))
    # x_valid_norm2 = (x_validation - x_validation.min(0)) / x_validation.ptp(0)
    # y_predicted = clf.predict_2(x_validation)
    # accuracy = (y_validation == y_predicted).mean()
    # print(accuracy)

    pred_y = clf.predict(test_x)

    # Calculate accuracy, precision, recall and confusion matrix
    print('Test Accuracy: ', accuracy_score(test_y, pred_y))
    print('Test Precision: ', precision_score(test_y, pred_y, zero_division=0, average='weighted'))
    print('Test Recall: ', recall_score(test_y, pred_y, average='weighted'))
    print('Test f1', f1_score(test_y, pred_y, zero_division=1, average='weighted', labels=np.unique(pred_y)))
    cm = confusion_matrix(test_y, pred_y)
    print('Confustion Matrix', cm)
    cmd = ConfusionMatrixDisplay(confusion_matrix=cm)
    cmd.plot()
    plt.savefig('GDA_Confusion', dpi=300)
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
    plt.title('GDA ROC curve')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive rate')
    plt.legend(loc='best')
    plt.savefig('GDA ROC', dpi=300)

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

class GDA:
    """Gaussian Discriminant Analysis.

    Example usage:
        > clf = GDA()
        > clf.fit(x_train, y_train)
        > clf.predict(x_eval)
    """
    def __init__(self, eps=1e-10, verbose=True):
        """
        Initialization of class.
        Args:
            eps: Threshold for determining convergence.
            verbose: Print loss values during training.
        """
        self.eps = eps
        self.verbose = verbose

    def fit(self, x, y):
        """Run solver to fit Guassian GDA model. Sets theta for prediction.

        Args:
            x: Training example inputs. Shape (n_examples, dim).
            y: Training example labels. Shape (n_examples,).

        """
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
        """
        Compute shared sigma.
        Args:
            x: Training example inputs.
            y: Training example labels.

        Returns:
            Sigma array.
        """
        x_minus_mu = x.copy().astype('float64')
        for i in range(len(self.mu)):
            x_minus_mu[y==self.y_classes[i]] -= self.mu[i]
        return (1 / len(y)) * x_minus_mu.T.dot(x_minus_mu)

    def predict(self, X):
        """
        Returns the prediction for each row of examples X.
        Args:
            X: Training example inputs.

        Returns:
            Array of probabilities of each class for the respective sample.
        """
        return np.apply_along_axis(self.get_prob, 1, X)

    def score(self, X, y):
        return (self.predict(X) == y).mean()

    def predict_2(self, x):
        """
        Predictions of probabilities using self.theta.
        It raises overflow problems.
        Args:
            x: Training example inputs.

        Returns:
            Array of probabilities of each class for the respective example.
        """
        print(np.finfo(np.double))
        new_x = np.empty((x.shape[0], self.theta.shape[1]), dtype=np.double)
        new_x = np.longdouble(np.dot(x.astype(np.longdouble), self.theta.astype(np.longdouble)))
        max_value = np.max(new_x, axis=1)[:, np.newaxis]
        new_x = new_x - max_value
        new_x = np.exp(new_x)
        sums = np.sum(new_x, axis=1)[:, np.newaxis]
        p = new_x / sums
        return new_x.argmax(axis=1)

    def get_prob(self, x):
        """
        Get the probabilites for each class individually (for one sample only)
        Args:
            x: values for the specific row example

        Returns:
            One-hot representation of the predicted class (max argument).
        """
        new_p = np.exp(-0.5 * np.sum((x - self.mu).dot(self.sigma_inv) * (x - self.mu), axis=1)) * self.phi
        sum = np.sum(new_p)
        p = new_p / sum
        return np.argmax(p)

if __name__ == '__main__':
    main(image_path='images.csv')

