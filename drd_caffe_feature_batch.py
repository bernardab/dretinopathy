"""
used caffenet as fixed feature extractor
http://nbviewer.ipython.org/github/BVLC/caffe/blob/master/examples/filter_visualization.ipynb

process
1. 
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

def get_blob_data(img_fname_list, net, transformer):
    """ compute a list of net inputs using the specified images and caffe model details.

    :param img_fname_list: list of input image filename
    :param net: caffe model net
    :param transformer: input processing object
    :returns:  list of blob data
    :rtype: dict

    """

    blob_list = []
    for img_fname in img_fname_list:
        net.blobs['data'].reshape(1,3,227,227)
        net.blobs['data'].data[...] = transformer.preprocess('data', caffe.io.load_image(img_fname))
        blob_list.append(net.blobs['data'].data)
        
    return blob_list

    
def extract_layer(net, layer):
    """Extract output features and filters for the specified layer

    :param net: forward net object
    :param layer: layer name
    :returns: a tuple ofoutput features and filters
    :rtype:  tuple

    usage example:  extract_layer(caffenet_dir,'fc6')

    """

    fea = net.blobs[layer].data
    filters = net.params[layer][0].data

    return fea, filters


def vis_square(data, padsize=1, padval=0):
    """
    take an array of shape (n, height, width) or (n, height, width, channels)
    and visualize each (height, width) thing in a grid of size approx. sqrt(n) by sqrt(n)

    """
    data -= data.min()
    data /= data.max()
    
    # force the number of filters to be square
    n = int(np.ceil(np.sqrt(data.shape[0])))
    padding = ((0, n ** 2 - data.shape[0]), (0, padsize), (0, padsize)) + ((0, 0),) * (data.ndim - 3)
    data = np.pad(data, padding, mode='constant', constant_values=(padval, padval))
    
    # tile the filters into an image
    data = data.reshape((n, n) + data.shape[1:]).transpose((0, 2, 1, 3) + tuple(range(4, data.ndim + 1)))
    data = data.reshape((n * data.shape[1], n * data.shape[3]) + data.shape[4:])
    
    plt.imshow(data)

    
if __name__ == '__main__':

    #load caffenet  model
    caffenet_model_file = caffe_root + '../../models/bvlc_reference_caffenet/deploy.prototxt'
    caffenet_pretrained_model = caffe_root + '../../models/bvlc_reference_caffenet/bvlc_reference_caffenet.caffemodel'
    ilsvrc_2012_mean_file = caffe_root + 'python/caffe/imagenet/ilsvrc_2012_mean.npy'
    net, transformer = load_model(caffenet_model_file, caffenet_pretrained_model, ilsvrc_2012_mean_file)

    # save caffenet features 
    trainLabels = bz.Data(data_dir + 'diabetic_ret/trainLabels.csv')
    results_dir= data_dir + 'diabetic_ret/results/'
    if not os.path.exists(results_dir):
        os.makedirs(results_dir)

    train_fnames = [data_dir + 'diabetic_ret/train/' + img_name + '.jpeg' for img_name in trainLabels['image']] 
    
    import pdb
    pdb.set_trace()
    
    # perform forwrd inference in batch mode
    blob_data = get_blob_data(train_fnames, net, transformer)
    net.forward_all(data=blob_data, blobs=['fc6', 'fc7', 'fc8'])

    #np.save(results_dir + 'fc6_caffenet.npy', np.array(fc6_stack))
    #np.save(results_dir + 'fc7_caffenet.npy', np.array(fc7_stack))
    #np.save(results_dir + 'fc8_caffenet.npy', np.array(fc8_stack))
