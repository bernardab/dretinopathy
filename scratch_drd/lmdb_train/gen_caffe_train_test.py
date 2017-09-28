# generate train and test(i.e validation) files for caffe finetuning

import os
import sys
import numpy as np
import blaze as bz
import glob
import random

data_dir = os.environ['DATA_DIR']

# train and test images and lable csvs
train_csv = bz.Data(data_dir + 'diabetic_ret/trainLabels.csv')

train = []
for img_name, label in train_csv:
    #train.append(data_dir + 'diabetic_ret/train_resized/' + img_name + '.jpeg ' + str(label))
    train.append(img_name + '.jpeg ' + str(label))
random.shuffle(train)

bz.odo(train[len(train)/5:], 'train.txt', sep='/n')
bz.odo(train[:len(train)/5], 'val.txt', sep='/n')
