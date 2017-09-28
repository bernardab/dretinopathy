# train drd from scratch tryout 

import numpy as np
import os
import sys

caffe_root = os.environ['CAFFE_ROOT']
sys.path.insert(0, caffe_root + 'python')
import caffe

caffe.set_device(0)
caffe.set_mode_gpu()


niter = 2000
# losses will also be stored in the log
train_loss = np.zeros(niter)

# We create a solver that fine-tunes from a previously trained network.
solver = caffe.SGDSolver('models/solver.prototxt')

# We run the solver for niter times, and record the training loss.
for it in range(niter):
    solver.step(1)  # SGD by Caffe

    # store the train loss
    train_loss[it] = solver.net.blobs['loss3/loss3'].data
    print 'iter %d, finetune_loss=%f' % (it, train_loss[it])

    # run test 
    if it % 10 == 0:

        test_iters = 10
        accuracy = 0
        for t in np.arange(test_iters):
            solver.test_nets[0].forward()
            accuracy += solver.test_nets[0].blobs['loss3/top-1'].data

        accuracy /= test_iters
        print 'Accuracy for fine-tuning:', accuracy

