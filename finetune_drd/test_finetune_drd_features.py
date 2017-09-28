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
#caffe.set_mode_cpu()

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
    model_file = 'models/deploy.prototxt'
    pretrained_model = 'models/diabetic_ret_finetune_iter_10000.caffemodel'
    ilsvrc_2012_mean_file = caffe_root + 'python/caffe/imagenet/ilsvrc_2012_mean.npy'
    net, transformer = load_model(model_file, pretrained_model, ilsvrc_2012_mean_file)

    # save caffenet features 
    train_csv = bz.Data(data_dir + 'diabetic_ret/trainLabels.csv')
    test_csv = bz.Data(data_dir + 'diabetic_ret/sampleSubmission.csv')
    results_dir= data_dir + 'diabetic_ret/results/'
    if not os.path.exists(results_dir):
        os.makedirs(results_dir)

    # extract features for validation and linear classifying on training set 
    # also exports features
    val_features = False
    if val_features:
        fc7_stack = []
        prob_stack = []
        class_stack = []
        correct = 0
        accuracy = 0
        for img_name, label in train_csv:
            img = data_dir + 'diabetic_ret/train/' + img_name + '.jpeg'
            forward_pass(img, net, transformer)
            fc7 = net.blobs['fc7'].data[0]
            prob = net.blobs['prob'].data[0]
            cls = net.blobs['prob'].data[0].argmax()
            fc7_stack.append(fc7.copy())
            prob_stack.append(prob.copy())
            class_stack.append(cls.copy())

            if cls==label:
                correct+=1
                accuracy = float(correct)/len(class_stack)
            print 'train ', img_name, ' expected', label, 'predicted ', cls, 'accuracy ', accuracy

        np.save(results_dir + 'fc7_drd_train.npy', np.array(fc7_stack))
        np.save(results_dir + 'prob_drd_train.npy', np.array(prob_stack))
        np.save(results_dir + 'class_drd_train.npy', np.array(class_stack))

    fc7_stack = []
    prob_stack = []
    class_stack = []
    for img_name in test_csv['image']:
        img = data_dir + 'diabetic_ret/test/' + img_name + '.jpeg'
        forward_pass(img, net, transformer)
        fc7 = net.blobs['fc7'].data[0]
        prob = net.blobs['prob'].data[0]
        cls = net.blobs['prob'].data[0].argmax()
        fc7_stack.append(fc7.copy())
        prob_stack.append(prob.copy())
        class_stack.append(cls.copy())
        print 'test ', img_name, ' ', cls

    np.save(results_dir + 'fc7_drd_test.npy', np.array(fc7_stack))
    np.save(results_dir + 'prob_drd_test.npy', np.array(prob_stack))
    np.save(results_dir + 'class_drd_test.npy', np.array(class_stack))

    # save test lables for kaggle submision
    test_result = results_dir + 'class_drd_test.csv'
    test_imgs = np.array(bz.Data(data_dir + 'diabetic_ret/sampleSubmission.csv')['image'])
    test_labels_df = bz.Data(zip(test_imgs, np.int64(class_stack)), fields=['image','level'])
    bz.odo(test_labels_df, test_result)
