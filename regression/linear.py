import os
import random
import scipy.stats as stats
import pylab
import util
import numpy as np
from PIL import Image, ImageStat
import skimage.measure
from skimage.feature import greycomatrix, greycoprops
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score

class LinearModel(object):
    """Base class for linear models."""

    def __init__(self, theta=None):
        """
        Args:
            theta: Weights vector for the model.
        """
        self.theta = theta

    def fit(self, X, y):
        """Run solver to fit linear model. You have to update the value of
        self.theta using the normal equations.

        Args:
            X: Training example inputs. Shape (n_examples, dim).
            y: Training example labels. Shape (n_examples,).
        """
        inverse = np.linalg.pinv(np.dot(X.T, X))
        self.theta = inverse.dot(np.dot(X.T, y))
        # self.theta = np.linalg.solve(np.dot(X.T, X), np.dot(X.T, y))

    def predict(self, X):
        """
        Make a prediction given new inputs x.
        Returns the numpy array of the predictions.

        Args:
            X: Inputs of shape (n_examples, dim).

        Returns:
            Outputs of shape (n_examples,).
        """
        # *** START CODE HERE ***
        return np.dot(X, self.theta)
        # *** END CODE HERE ***

def createFileList(myDir, format='.jpg'):
    """Create file list to load images,

    Args:
        myDir: Folder where the image are.
        format: Type of files to generate the list.

    Returns:
        List of files.
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
    """"Get labels from labele.csv file

    Returns:
        Numpy array of labels (n_examples,).
    """
    train_labels = np.loadtxt('labels.csv', delimiter=',')
    return train_labels

def get_images_variables():
    """"Get variables from images files
    mean RGB, median RGB, redness index, colouration index, hue index, saturation index,
    mean HSV, median HSV, mean gray, median gray, entropy, contrast, energy, homogeneity

    Returns:
        Numpy array of variables (n_examples, 22).
    """
    path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'images'))
    myFileList = createFileList(path)
    myFileList = sorted(myFileList)
    train_data = np.array([])
    for file in myFileList:
        img_file = Image.open(file)
        RGB_pixels = np.asarray(img_file.getdata(), dtype=np.int)
        # Add RGB mean values at feature vector
        image_features = np.mean(RGB_pixels, axis = 0)
        mean_red = image_features[0]
        mean_green = image_features[1]
        mean_blue = image_features[2]
        # Add RGB median values at feature vector
        image_features = np.append(image_features, np.median(RGB_pixels, axis = 0))
        # Redness index
        if (mean_blue * mean_green ** 3) != 0:
            redness_index = (mean_red ** 2) / (mean_blue * mean_green ** 3)
        else:
            redness_index = mean_red ** 2
        image_features = np.append(image_features, [redness_index])
        # Colouration index
        if (mean_red + mean_green) != 0:
            colouration_index = (mean_red - mean_green) / (mean_red + mean_green)
        else:
            colouration_index = mean_red - mean_green
        image_features = np.append(image_features, [colouration_index])
        # Hue index
        if (mean_green - mean_blue) != 0:
            hue_index = (2 * mean_red - mean_green - mean_blue) / (mean_green - mean_blue)
        else:
            hue_index = (2 * mean_red - mean_green - mean_blue)
        image_features = np.append(image_features, [hue_index])
        # Saturation index
        if (mean_red + mean_blue) != 0:
            saturation_index = (mean_red - mean_blue) / (mean_red + mean_blue)
        else:
            saturation_index = (mean_red - mean_blue)
        image_features = np.append(image_features, [saturation_index])
        # HSV
        hsv_image = img_file.convert('HSV')
        HSV_pixels = np.asarray(hsv_image.getdata(), dtype=np.int)
        # Add HSV mean values at feature vector
        image_features = np.append(image_features, np.mean(HSV_pixels, axis = 0))
        # Add HSV median values at feature vector
        image_features = np.append(image_features, np.median(HSV_pixels, axis=0))
        # Grayscale image
        grayscale_image = img_file.convert('L')
        grayscale_pixels = np.asarray(grayscale_image.getdata(), dtype=np.int)
        # Mean Gray
        image_features = np.append(image_features, np.mean(grayscale_pixels, axis=0))
        # Median Gray
        image_features = np.append(image_features, np.median(grayscale_pixels, axis=0))
        # Entropy
        entropy = skimage.measure.shannon_entropy(grayscale_image)
        image_features = np.append(image_features, [entropy])
        # Contrast
        contrast = ImageStat.Stat(grayscale_image).stddev
        image_features = np.append(image_features, contrast)
        glcm = greycomatrix(np.array(grayscale_image), [1], [0], symmetric=True, normed=True)
        # Energy and homogeneity
        homogeneity = greycoprops(glcm, 'homogeneity')[0,0]
        energy = greycoprops(glcm, 'energy')[0,0]
        image_features = np.append(image_features, [energy])
        image_features = np.append(image_features, [homogeneity])
        if train_data.size == 0:
            train_data = image_features.reshape(22,1)
        else:
            train_data = np.append(train_data, image_features.reshape(22,1), axis=1)
        print(train_data.T.shape)
    return train_data.T

def gaussian_variables_analysis(X):
    """
    Analyze if each variable is a normal distribution or not,
    It also saves QQ_plots of each variable,

    Args:
        X: Inputs of shape (n_examples, dim).

    """
    n_variables = X.shape[1]
    for i in range(n_variables):
        pylab.clf()
        pylab.cla()
        stats.probplot(X[:,i], dist="norm", plot=pylab)
        # shapiro_test  = stats.shapiro(X[:,i])
        stat, p = stats.normaltest(X[:,i])
        alpha = 0.05
        if p > alpha:
            print('Feature ' + str(i+1) + ' looks Gaussian; p =  ' + str(p))
        else:
            print('Feature ' + str(i + 1) + ' does not look Gaussian; p =  ' + str(p))
        pylab.savefig('./QQ_plots/x_'+str(i+1))

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

def run_exp():
    random.seed(0)
    # Import variables from images
    # orig_y = get_labels()
    # orig_x = get_images_variables()
    orig_x, orig_y = util.load_dataset('images.csv', add_intercept=True)
    # Create the training sample
    train_x, val_test_x, train_y, val_test_y = train_test_split(orig_x, orig_y, test_size=0.3, random_state=1)
    # Split the remaining observations into validation and test
    val_x, test_x, val_y, test_y = train_test_split(val_test_x, val_test_y, test_size=0.5, random_state=1)

    # Create the linear model and predict the values
    lr = LinearModel()
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

    # Randomly choose 10 inputs without replacement
    input_indices = random.sample(range(0, train_x.shape[1]), 22)
    # Train the model using the training set
    lr.fit(train_x[:, input_indices], train_y)

    # For 1000 times, randomly choose 7 inputs, estimate regression,
    # and save if performance on validation is better than that of
    # previous regressions
    betterResult = False
    for j in range(0, 1000):
        print(j)
        lr_j = LinearModel()
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

    # Saving results and images on file
    #np.savetxt("images.csv", orig_x, delimiter=",")
    #np.savetxt("predicted_y.csv", predicted_y, delimiter=",")
    #np.savetxt("theta.csv", lm.theta, delimiter=",")
    # *** END CODE HERE ***

def main():
    '''
    Run all expetriments
    '''
    run_exp()

if __name__ == '__main__':
    main()
