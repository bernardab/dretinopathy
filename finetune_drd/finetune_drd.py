# finetune caffenet with ddiabetic ret data

import numpy as np
import os
import sys

caffe_root = os.environ['CAFFE_ROOT']
sys.path.insert(0, caffe_root + 'python')
import caffe

caffe.set_mode_cpu()

caffenet_pretrained_model = caffe_root + '../../models/bvlc_reference_caffenet/bvlc_reference_caffenet.caffemodel'

niter = 10
# losses will also be stored in the log
train_loss = np.zeros(niter)
scratch_train_loss = np.zeros(niter)

# We create a solver that fine-tunes from a previously trained network.
solver = caffe.SGDSolver('models/solver.prototxt')
solver.net.copy_from(caffenet_pretrained_model)

# For reference, we also create a solver that does no finetuning.
scratch_solver = caffe.SGDSolver('models/solver.prototxt')

# We run the solver for niter times, and record the training loss.
for it in range(niter):
    solver.step(1)  # SGD by Caffe
    scratch_solver.step(1)
    # store the train loss
    train_loss[it] = solver.net.blobs['loss'].data
    scratch_train_loss[it] = scratch_solver.net.blobs['loss'].data
    if it % 10 == 0:
        print 'iter %d, finetune_loss=%f, scratch_loss=%f' % (it, train_loss[it], scratch_train_loss[it])
print 'done'


# run test
test_iters = 10
accuracy = 0
scratch_accuracy = 0
for it in arange(test_iters):
    solver.test_nets[0].forward()
    accuracy += solver.test_nets[0].blobs['accuracy'].data
    scratch_solver.test_nets[0].forward()
    scratch_accuracy += scratch_solver.test_nets[0].blobs['accuracy'].data
accuracy /= test_iters
scratch_accuracy /= test_iters
print 'Accuracy for fine-tuning:', accuracy
print 'Accuracy for training from scratch:', scratch_accuracy
