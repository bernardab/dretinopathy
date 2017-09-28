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

caffe.set_device(0)
caffe.set_mode_gpu()

def load_model(model_file, pretrained_model, mean_values):
    """load the net in the test phase for inference, and configure input preprocessing.

    :param model_file: .prototxt file path
    :param pretrained_model: pretrained model path 
    :param mean_values: mean pixels to subtract from input
    :returns: a tuple containing the model net and input preprocessing object
    :rtype: tuple

    """
    
    net = caffe.Net(model_file, pretrained_model, caffe.TEST)

    # input preprocessing: 'data' is the name of the input blob == net.inputs[0]
    transformer = caffe.io.Transformer({'data': net.blobs['data'].data.shape})
    transformer.set_transpose('data', (2,0,1))
    transformer.set_mean('data', mean_values) # mean pixels
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

    net.blobs['data'].reshape(1,3,224,224)
    net.blobs['data'].data[...] = transformer.preprocess('data', caffe.io.load_image(img_fname))
    out = net.forward()

    return out

    
if __name__ == '__main__':
    
    results_dir= data_dir + 'diabetic_ret/results/'
    if not os.path.exists(results_dir):
        os.makedirs(results_dir)

    #load caffenet  model
    #model_file = 'models/deploy.prototxt'
    #pretrained_model = 'models/diabetic_ret_finetune_iter_10000.caffemodel'
    #ilsvrc_2012_mean_file = caffe_root + 'python/caffe/imagenet/ilsvrc_2012_mean.npy'

    #googlenet model
    model_file = 'gnet_models_resized/deploy.prototxt'
    #pretrained_model = 'gnet_models/diabetic_ret_finetune_iter_40000.caffemodel'
    pretrained_model = 'gnet_models_resized/_iter_225000.caffemodel'
    mean_values = np.array([104, 117, 123])
    test_result = results_dir + 'class_drd_test_gnet.csv'
    
    net, transformer = load_model(model_file, pretrained_model, mean_values)

    # save caffenet features 
    train_csv = bz.Data(data_dir + 'diabetic_ret/trainLabels.csv')
    test_csv = bz.Data(data_dir + 'diabetic_ret/sampleSubmission.csv')

    # extract features for validation and linear classifying on training set 
    # also exports features
    val_features = True
    if val_features:
        class_stack = []
        correct = 0
        accuracy = 0
        for img_name, label in train_csv:
            img = data_dir + 'diabetic_ret/train_resized/' + img_name + '.jpeg'
            forward_pass(img, net, transformer)
            cls = net.blobs['prob'].data[0].argmax()
            class_stack.append(cls.copy())

            if cls==label:
                correct+=1
                accuracy = float(correct)/len(class_stack)
            print 'train ', img_name, ' expected', label, 'predicted ', cls, 'accuracy ', accuracy

    class_stack = []
    for img_name in test_csv['image']:
        img = data_dir + 'diabetic_ret/test_resized/' + img_name + '.jpeg'
        forward_pass(img, net, transformer)
        cls = net.blobs['prob'].data[0].argmax()
        class_stack.append(cls.copy())
        print 'test ', img_name, ' ', cls

    # save test lables for kaggle submision
    test_imgs = np.array(bz.Data(data_dir + 'diabetic_ret/sampleSubmission.csv')['image'])
    test_labels_df = bz.Data(zip(test_imgs, np.int64(class_stack)), fields=['image','level'])
    bz.odo(test_labels_df, test_result)
