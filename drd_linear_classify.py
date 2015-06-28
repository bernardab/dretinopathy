"""
classify caffe features using linear classifiers
"""

import numpy as np
import blaze as bz
import os
import glob
from sklearn import cross_validation
from sklearn import svm

data_dir = os.environ['DATA_DIR']
results_dir= data_dir + 'diabetic_ret/results/'

if __name__ == '__main__':

    # load features and labels
    labels = list(bz.Data(data_dir + 'diabetic_ret/trainLabels.csv')['level'])
    features = np.load(results_dir + 'fc7_caffenet.npy')

    # svm classification with cross validation
    clf = svm.SVC(kernel='linear', C=1)
    scores = cross_validation.cross_val_score(clf, features, labels, cv=5)

    # predict test_set and save as csv
    clf.predict(test_features)
    
