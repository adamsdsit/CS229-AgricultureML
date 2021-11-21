import numpy as np
import math
import os
import scipy.stats as stats
import pylab
from PIL import Image, ImageStat
import skimage.measure
from skimage.feature import greycomatrix, greycoprops

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
        """Run solver to fit linear model. You have to update the value of
        self.theta using the normal equations.

        Args:
            X: Training example inputs. Shape (n_examples, dim).
            y: Training example labels. Shape (n_examples,).
        """
        # *** START CODE HERE ***
        self.theta = np.linalg.solve(np.dot(X.T, X), np.dot(X.T, y))
        # *** END CODE HERE ***

    def createFileList(self, myDir, format='.jpg'):
        fileList = []
        print(myDir)
        for root, dirs, files in os.walk(myDir, topdown=False):
            for name in files:
                if name.endswith(format):
                    fullName = os.path.join(root, name)
                    fileList.append(fullName)
        return fileList

    def get_train_y(self):
        train_labels = np.loadtxt('train_labels_only.csv', delimiter=',')
        return train_labels

    def get_test_y(self):
        test_labels = np.loadtxt('test_labels_only.csv', delimiter=',')
        return test_labels

    def get_train_x(self):
        myFileList = self.createFileList('./train_images/')
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
            else:                redness_index = mean_red ** 2
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

    def get_test_x(self):
        myFileList = self.createFileList('./test_images/')
        myFileList = sorted(myFileList)
        test_data = np.array([])
        for file in myFileList:
            img_file = Image.open(file)
            RGB_pixels = np.asarray(img_file.getdata(), dtype=np.int)
            # Add RGB mean values at feature vector
            image_features = np.mean(RGB_pixels, axis=0)
            mean_red = image_features[0]
            mean_green = image_features[1]
            mean_blue = image_features[2]
            # Add RGB median values at feature vector
            image_features = np.append(image_features, np.median(RGB_pixels, axis=0))
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
            image_features = np.append(image_features, np.mean(HSV_pixels, axis=0))
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
            homogeneity = greycoprops(glcm, 'homogeneity')[0, 0]
            energy = greycoprops(glcm, 'energy')[0, 0]
            image_features = np.append(image_features, [energy])
            image_features = np.append(image_features, [homogeneity])
            if test_data.size == 0:
                test_data = image_features.reshape(22, 1)
            else:
                test_data = np.append(test_data, image_features.reshape(22, 1), axis=1)
            print(test_data.T.shape)
        return test_data.T

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

def ceil(x):
    return math.ceil(x/0.7)

def run_exp():
    # *** START CODE HERE ***
    lm = LinearModel()
    train_y = lm.get_train_y()
    train_x = lm.get_train_x()

    for i in range(22):
        if i == 0:
            print(train_x[:,i])
        pylab.clf()
        pylab.cla()
        stats.probplot(train_x[:,i], dist="norm", plot=pylab)
        shapiro_test  = stats.shapiro(train_x[:,i])
        stat, p = stats.normaltest(train_x[:,i])
        alpha = 0.05
        if p > alpha:
            print('Feature ' + str(i+1) + ' looks Gaussian; p =  ' + str(p))
        else:
            print('Feature ' + str(i + 1) + ' does not look Gaussian; p =  ' + str(p))
        pylab.savefig('x_'+str(i+1))

    lm.fit(train_x, train_y)
    test_x = lm.get_test_x()
    test_y = lm.get_test_y()
    predicted_y = lm.predict(test_x)
    rmsd = math.sqrt(np.square(test_y-predicted_y).sum()/455)
    print(rmsd)
    ceil_v = np.vectorize(ceil)
    accuracy = (ceil_v(test_y) == ceil_v(predicted_y)).sum() / predicted_y.shape[0]
    print(accuracy)
    np.savetxt("train_x_matrix.csv", train_x, delimiter=",")
    np.savetxt("test_x_matrix.csv", train_x, delimiter=",")
    np.savetxt("predicted_y.csv", predicted_y, delimiter=",")
    np.savetxt("theta.csv", lm.theta, delimiter=",")
    # *** END CODE HERE ***

def main(train_path, small_path, eval_path):
    '''
    Run all expetriments
    '''
    # *** START CODE HERE ***
    run_exp()
    # *** END CODE HERE ***

if __name__ == '__main__':
    main(train_path='train.csv',
        small_path='small.csv',
        eval_path='test.csv')
