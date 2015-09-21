import numpy as np
import PIL.Image

caffe_root = '/works/caffe/'
import sys; sys.path.insert(0, caffe_root + 'python')
import caffe; caffe.set_mode_cpu()

# Load the original network and extract the fully connected layers' parameters.
MODEL_DEPLOY_FILENAME = caffe_root + 'models/bvlc_googlenet/deploy.prototxt'
MODEL_WEIGHT_FILENAME = caffe_root + 'models/bvlc_googlenet/bvlc_googlenet.caffemodel'
net = caffe.Net(MODEL_DEPLOY_FILENAME, MODEL_WEIGHT_FILENAME, caffe.TEST)
params = ['loss3/classifier']
# fc_params = {name: (weights, biases)}
fc_params = {pr: (net.params[pr][0].data, net.params[pr][1].data) for pr in params}

for fc in params:
  print '{} weights are {} dimensional and biases are {} dimensional'.format(fc, fc_params[fc][0].shape, fc_params[fc][1].shape)

import pdb; pdb.set_trace()
# Load the fully convolutional network to transplant the parameters.
MODEL_FCN_DEPLOY_FILENAME = caffe_root + 'models/bvlc_googlenet/deploy_fcn.prototxt'
MODEL_FCN_WEIGHT_FILENAME = caffe_root + 'models/bvlc_googlenet/bvlc_googlenet.caffemodel'
net_full_conv = caffe.Net(MODEL_FCN_DEPLOY_FILENAME, MODEL_FCN_WEIGHT_FILENAME, caffe.TEST)
params_full_conv = ['loss3/classifier']
# conv_params = {name: (weights, biases)}
conv_params = {pr: (net_full_conv.params[pr][0].data, net_full_conv.params[pr][1].data) for pr in params_full_conv}

for conv in params_full_conv:
  print '{} weights are {} dimensional and biases are {} dimensional'.format(conv, conv_params[conv][0].shape, conv_params[conv][1].shape)
