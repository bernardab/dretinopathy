"""
classify caffe features using linear classifiers
"""

import numpy as np
import blaze as bz
import os
import glob
import xgboost as xgb
from sklearn.cross_validation import train_test_split
from sklearn.metrics import accuracy_score

data_dir = os.environ['DATA_DIR']
results_dir = data_dir + 'diabetic_ret/results/'
test_result = results_dir + 'fea_finetune_fc8_xgboost_clf.csv'

if __name__ == '__main__':

    # load features and labels
    train_features = np.load(results_dir + 'fc8_drd_train.npy')
    train_labels = np.array(bz.Data(data_dir + 'diabetic_ret/trainLabels.csv')['level'])
    #test_features = np.load(results_dir + 'fc8_drd_test.npy')
    test_imgs = np.array(bz.Data(data_dir + 'diabetic_ret/sampleSubmission.csv')['image'])

    # split train features into train and validation sets
    train_X, val_X, train_Y, val_Y = train_test_split(train_features, train_labels, test_size=0.33, random_state=42)
    
    # train and test images
    xg_train = xgb.DMatrix(train_X, label=train_Y)
    xg_val = xgb.DMatrix(val_X, val_Y)

    # setup parameters for xgboost
    param = {}
    # use softmax multi-class classification
    param['objective'] = 'multi:softmax'
    # scale weight of positive examples
    param['eta'] = 0.1
    param['max_depth'] = 6
    param['silent'] = 1
    param['nthread'] = 4
    param['num_class'] = 5


    num_round = 5
    bst = xgb.train(param, xg_train, num_round);

    # get prediction
    val_pred = bst.predict( xg_val );

    # get metrics
    score = accuracy_score(val_Y, val_pred)

    # predict test_set and save as csv
    run_test = True
    if run_test:
        xg_test = xgb.DMatrix(test_features)
        test_labels = bst.predict(xg_test)
        test_labels_df = bz.Data(zip(test_imgs, np.int64(test_labels)), fields=['image','level'])
        bz.odo(test_labels_df, test_result)
