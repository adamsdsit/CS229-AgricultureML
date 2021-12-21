import numpy as np
import random
import util
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import cross_val_predict
from sklearn.model_selection import KFold
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import cross_val_score

class GradientBoostingClass(object):
    """Base class for Gradient Boosting model."""

    def __init__(self, n=None, regressor=None):
        """
            Args:
                regressor: nklearn.ensemble.GradientBoostingRegressor model.
            """
        self.regressor = regressor
        self.n = n

    def fit(self, X, y):
        """Run solver to fit Gradient Boosting Regressor model.

            Args:
                X: Training example inputs. Shape (n_examples, dim).
                y: Training example labels. Shape (n_examples,).
        """
        # {'learning_rate': 0.01, 'max_depth': 4, 'n_estimators': 2000, 'random_state': 1, 'subsample': 0.75}
        if (self.n != None):
            self.regressor = GradientBoostingRegressor(
                max_depth=6,
                n_estimators=500,
                learning_rate=.01,
                random_state=2,
                subsample=0.4,
                min_samples_split=2,
                min_samples_leaf=1)
        else:
            self.regressor = GradientBoostingRegressor()
        self.regressor.fit(X, y)

    def predict(self, X):
        """
            Make a prediction given new inputs x.
            Returns the numpy array of the predictions.

            Args:
                X: Inputs of shape (n_examples, dim).

            Returns:
                Outputs of shape (n_examples,).
        """
        return self.regressor.predict(X)

def get_models():
    models = dict()
    n_trees = [10, 50, 100, 500, 1000, 5000]
    for n in n_trees:
        models[str(n)] = GradientBoostingClass(n)
    return models

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
    # Establish a random seed
    random.seed(0)
    # Read the file of variables
    orig_x, orig_y = util.load_dataset(image_path, add_intercept=False)
    # Create the training sample
    # WE're using cross-validation instead of splitting training and validation sets
    train_x, test_x, train_y, test_y = train_test_split(orig_x, orig_y, test_size=0.15, random_state=1)
    # val_x, test_x, val_y, test_y = train_test_split(val_test_x, val_test_y, test_size=0.5, random_state=1)

    gb = GradientBoostingClass(2000)
    gb.fit(train_x, train_y)
    train_y_pred = gb.predict(train_x)
    test_y_pred = gb.predict(test_x)

    print('MSE on training set:')
    print(mean_squared_error(train_y, train_y_pred))

    print('MSE on testing set:')
    print(mean_squared_error(test_y, test_y_pred))

    print('R squared on training set:')
    print(r2_score(train_y, train_y_pred))
    print('R squared on test set:')
    print(r2_score(test_y, test_y_pred))

    # Measure accuracy for classification problem
    print('Accuracy: ')
    accuracy = (classification(orig_y, test_y) == classification(orig_y, test_y_pred)).sum() / test_y.shape[0]
    print(accuracy)

    # Tuning Gradient Boosting
    GBR = GradientBoostingRegressor()
    search_grid = {'n_estimators': [100, 500, 1000], 'learning_rate': [0.01, .1, 1], 'max_depth': [2,4,10],
                   'subsample': [0.5, 0.75, 1], 'random_state': [1]}
    cv = KFold(n_splits=29, shuffle=True, random_state=1)
    search = GridSearchCV(estimator=GBR, param_grid=search_grid, scoring='neg_mean_squared_error', n_jobs=1,
                          cv=cv)
    search.fit(train_x, train_y)
    pred_y = search.predict(test_x)

    # prepare the cross-validation procedure
    # define folds to test
    folds = range(2, 31)
    # record mean and min/max of each set of results
    means, mins, maxs = list(), list(), list()
    # evaluate each k value
    best_k = folds[0]
    best_error = float('-inf')
    for k in folds:
        GBR = GradientBoostingRegressor()
        # define the test condition
        cv = KFold(n_splits=k, shuffle=True, random_state=1)
        # evaluate k value
        scores = cross_val_score(GBR, train_x, train_y, scoring='neg_mean_squared_error', cv=cv, n_jobs=-1)
        pred_Y_cross = cross_val_predict(GBR, train_x, train_y, cv=cv)
        accuracy = (classification(orig_y, train_y) == classification(orig_y, pred_Y_cross)).sum() / train_y.shape[0]
        k_mean = np.mean(scores)
        k_min = scores.min()
        k_max = scores.max()
        if (k_mean > best_error):
            best_error = k_mean
            best_k = k
        # report performance
        print('> folds=%d, error=%.3f (%.3f,%.3f)' % (k, k_mean, k_min, k_max))
        print('> folds=%d, training accuracy=%.3f' % (k, accuracy))
        # store mean accuracy
        means.append(k_mean)
        # store min and max relative to the mean
        mins.append(k_mean - k_min)
        maxs.append(k_max - k_mean)

    # Calculate values for test_x
    cv = KFold(n_splits=best_k, shuffle=True, random_state=1)
    pred_Y_cross = cross_val_predict(GBR, test_x, test_y, cv=cv)

    print('MSE on test set - after cross validation:')
    print(mean_squared_error(test_y, pred_Y_cross))

    print('R squared on test set - after cross validation:')
    print(r2_score(test_y, pred_Y_cross))

    # Measure accuracy for classification problem
    print('Accuracy: ')
    print(best_k)
    accuracy = (classification(orig_y, test_y) == classification(orig_y, pred_Y_cross)).sum() / test_y.shape[0]
    print(accuracy)

def main(image_path):
    run_exp(image_path)

if __name__ == '__main__':
    main(image_path='images.csv')
