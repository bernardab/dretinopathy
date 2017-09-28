# train drd model from scratch

import os
import sys

caffe_root = os.environ['CAFFE_ROOT']
sys.path.insert(0, caffe_root + 'python')
import caffe

caffe.set_device(0)
caffe.set_mode_gpu()
#caffe.set_mode_cpu()

# We create a solver that fine-tunes from a previously trained network.
solver = caffe.SGDSolver('models/solver.prototxt')

# train model
solver.restore('models/_iter_475000.solverstate');
solver.solve()
