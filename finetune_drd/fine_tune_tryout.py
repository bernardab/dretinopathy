# finetune caffenet with ddiabetic ret data

import numpy as np
import os
import sys

caffe_root = os.environ['CAFFE_ROOT']
sys.path.insert(0, caffe_root + 'python')
import caffe

caffe.set_device(0)
caffe.set_mode_gpu()

pretrained_model = caffe_root + '../../models/bvlc_googlenet/bvlc_googlenet.caffemodel'

niter = 2000
# losses will also be stored in the log
train_loss = np.zeros(niter)

# We create a solver that fine-tunes from a previously trained network.
solver = caffe.SGDSolver('gnet_models/solver.prototxt')
solver.net.copy_from(pretrained_model)

# We run the solver for niter times, and record the training loss.
for it in range(niter):
    solver.step(1)  # SGD by Caffe

    # store the train loss
    train_loss[it] = solver.net.blobs['loss3/loss3'].data
    print 'iter %d, finetune_loss=%f' % (it, train_loss[it])

    # save model every 1000 iterations
    if it % 10 == 0:


        # save model
        if it % 1000 == 0:
            model_fname = 'gnet_models/diabetic_ret_finetune_' + str(it) + '.caffemodel'
            solver.net.save(model_fname)

        # run test
        test_iters = 10
        accuracy = 0
        for t in np.arange(test_iters):
            solver.test_nets[0].forward()
            accuracy += solver.test_nets[0].blobs['loss3/top-1'].data

        accuracy /= test_iters
        print 'Accuracy for fine-tuning:', accuracy

