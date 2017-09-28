"""
classify testing samples using finetuned drd model
"""

import os
import sys
import numpy as np
import blaze as bz
import glob

caffe_root = os.environ['CAFFE_ROOT']
data_dir = os.environ['DATA_DIR']
sys.path.insert(0, caffe_root + 'python')
import caffe
caffe.set_mode_cpu()


def load_model(model_file, pretrained_model, mean_file):
    """load the net in the test phase for inference, and configure input preprocessing.

    :param model_file: .prototxt file path
    :param pretrained_model: pretrained model path 
    :param mean_file: .npy mean file path 
    :returns: a tuple containing the model net and input preprocessing object
    :rtype: tuple

    """
    
    net = caffe.Net(model_file, pretrained_model, caffe.TEST)

    # input preprocessing: 'data' is the name of the input blob == net.inputs[0]
    transformer = caffe.io.Transformer({'data': net.blobs['data'].data.shape})
    transformer.set_transpose('data', (2,0,1))
    transformer.set_mean('data', np.load(mean_file).mean(1).mean(1)) # mean pixel
    transformer.set_raw_scale('data', 255)  # the reference model operates on images in [0,255] range instead of [0,1]
    transformer.set_channel_swap('data', (2,1,0))  # the reference model has channels in BGR order instead of RGB

    return net, transformer


def forward_pass(img_fname, net, transformer):
    """ perform foward inference on an image using the specified caffe model details.

    :param img_fname: input image filename
    :param net: caffe model net
    :param transformer: input processing object
    :returns: forward net
    :rtype: net object

    """

    net.blobs['data'].reshape(1,3,227,227)
    net.blobs['data'].data[...] = transformer.preprocess('data', caffe.io.load_image(img_fname))
    out = net.forward()

    return out

    
if __name__ == '__main__':

    #load caffenet  model
    caffenet_model_file = 'models/deploy.prototxt'
    caffenet_pretrained_model = 'models/diabetic_ret_finetune_0.caffemodel'
    ilsvrc_2012_mean_file = caffe_root + 'python/caffe/imagenet/ilsvrc_2012_mean.npy'
    net, transformer = load_model(caffenet_model_file, caffenet_pretrained_model, ilsvrc_2012_mean_file)

    # save caffenet features 
    train_csv = bz.Data(data_dir + 'diabetic_ret/trainLabels.csv')
    test_csv = bz.Data(data_dir + 'diabetic_ret/sampleSubmission.csv')
    results_dir= data_dir + 'diabetic_ret/results/'
    if not os.path.exists(results_dir):
        os.makedirs(results_dir)

    # check if model already exists here and load if it already exists
    fc7_stack = []
    prob_stack = []
    for img_name in train_csv['image']:
        img = data_dir + 'diabetic_ret/train/' + img_name + '.jpeg'
        forward_pass(img, net, transformer)
        fc7_stack.append(net.blobs['fc7'].data[0])
        prob_stack.append(net.blobs['prob'].data[0])
        print 'train ', img_name

    np.save(results_dir + 'fc7_drd_train.npy', np.array(fc7_stack))
    np.save(results_dir + 'prob_drd_train.npy', np.array(prob_stack))

    fc7_stack = []
    prob_stack = []
    for img_name in test_csv['image']:
        img = data_dir + 'diabetic_ret/test/' + img_name + '.jpeg'
        forward_pass(img, net, transformer)
        fc7_stack.append(net.blobs['fc7'].data[0])
        prob_stack.append(net.blobs['prob'].data[0])
        print 'test ', img_name

    np.save(results_dir + 'fc7_drd_test.npy', np.array(fc7_stack))
    np.save(results_dir + 'prob_drd_test.npy', np.array(prob_stack))
