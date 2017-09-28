from sklearn.metrics import accuracy_score
import os
import numpy as np
import blaze as bz

data_dir = os.environ['DATA_DIR']
results_dir= data_dir + 'diabetic_ret/results/'
train_csv = bz.Data(data_dir + 'diabetic_ret/trainLabels.csv')

expected = np.array(bz.Data(data_dir + 'diabetic_ret/trainLabels.csv')['level'])
prob = np.load(results_dir + 'prob_drd_train.npy')

predicted = []
[predicted.append(p.argmax() for p in prob)]

score = accuracy_score(expected, predicted)
