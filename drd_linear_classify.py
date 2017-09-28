"""
classify caffe features using linear classifiers
"""

import numpy as np
import blaze as bz
import os
import glob
from sklearn import cross_validation
from sklearn import svm
from sklearn import linear_model

data_dir = os.environ['DATA_DIR']
results_dir = data_dir + 'diabetic_ret/results/'
test_result = results_dir + 'fea_linear_clf.csv'

if __name__ == '__main__':

    # load features and labels
    train_features = np.load(results_dir + 'fc6_caffenet_train.npy')
    train_labels = list(bz.Data(data_dir + 'diabetic_ret/trainLabels.csv')['level'])
    test_features = np.load(results_dir + 'fc6_caffenet_test.npy')
    test_imgs = list(bz.Data(data_dir + 'diabetic_ret/sampleSubmission.csv')['image'])


    # get half of training featuers
    ns = train_features.shape[0]/2
    train_features = train_features[:ns]
    train_labels = train_labels[:ns]

    #  svm classifier evaluation with cross validation
    eval = 0
    clf = svm.SVC(kernel='linear', C=1)
    #clf = linear_model.SGDClassifier()
    if eval:
        scores = cross_validation.cross_val_score(clf, train_features, train_labels, cv=5)
    else:
        # fit classifier
        clf.fit(train_features, train_labels)
    
        # predict test_set and save as csv
        test_labels = clf.predict(test_features)
        test_labels_df = bz.Data(zip(test_imgs, test_labels), fields=['image','level'])
        

    
    
    
