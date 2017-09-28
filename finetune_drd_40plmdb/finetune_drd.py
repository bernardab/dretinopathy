# finetune caffenet with ddiabetic ret data

import os
import sys

caffe_root = os.environ['CAFFE_ROOT']
sys.path.insert(0, caffe_root + 'python')
import caffe

caffe.set_device(0)
caffe.set_mode_gpu()

#pretrained_model = caffe_root + '../../models/bvlc_reference_caffenet/bvlc_reference_caffenet.caffemodel'
pretrained_model = caffe_root + '../../models/bvlc_googlenet/bvlc_googlenet.caffemodel'

# We create a solver that fine-tunes from a previously trained network.
#solver = caffe.SGDSolver('models/solver.prototxt')
solver = caffe.SGDSolver('gnet_models/solver.prototxt')
solver.net.copy_from(pretrained_model)

# finetune model
#solver.solve() 
solver.restore('gnet_models/_iter_40000.solverstate');
solver.solve()
