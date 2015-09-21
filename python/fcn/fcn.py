import numpy as np
import PIL.Image

caffe_root = '/works/caffe/'
import sys; sys.path.insert(0, caffe_root + 'python')
import caffe; caffe.set_mode_cpu()

# Load the original network and extract the fully connected layers' parameters.
MODEL_DEPLOY_FILENAME = caffe_root + 'models/vgg/vgg_layer16_deploy.prototxt'
MODEL_WEIGHT_FILENAME = caffe_root + 'models/vgg/vgg_layer16.caffemodel'
net = caffe.Net(MODEL_DEPLOY_FILENAME, MODEL_WEIGHT_FILENAME, caffe.TEST)
params = ['fc6', 'fc7', 'fc8']
# fc_params = {name: (weights, biases)}
fc_params = {pr: (net.params[pr][0].data, net.params[pr][1].data) for pr in params}

for fc in params:
  print '{} weights are {} dimensional and biases are {} dimensional'.format(fc, fc_params[fc][0].shape, fc_params[fc][1].shape)

import pdb; pdb.set_trace()
# Load the fully convolutional network to transplant the parameters.
MODEL_FCN_DEPLOY_FILENAME = caffe_root + 'models/vgg/vgg_layer16_deploy_fcn.prototxt'
MODEL_FCN_WEIGHT_FILENAME = caffe_root + 'models/vgg/vgg_layer16.caffemodel'
net_full_conv = caffe.Net(MODEL_FCN_DEPLOY_FILENAME, MODEL_FCN_WEIGHT_FILENAME, caffe.TEST)
params_full_conv = ['fc6_conv', 'fc7_conv', 'fc8_conv']
# conv_params = {name: (weights, biases)}
conv_params = {pr: (net_full_conv.params[pr][0].data, net_full_conv.params[pr][1].data) for pr in params_full_conv}

for conv in params_full_conv:
  print '{} weights are {} dimensional and biases are {} dimensional'.format(conv, conv_params[conv][0].shape, conv_params[conv][1].shape)

import pdb; pdb.set_trace()
# transplant
for pr, pr_conv in zip(params, params_full_conv):
  conv_params[pr_conv][0].flat = fc_params[pr][0].flat  # flat unrolls the arrays
  conv_params[pr_conv][1][...] = fc_params[pr][1]

net_full_conv.save(caffe_root + 'models/vgg/vgg_layer16_fcn.caffemodel')

import pdb; pdb.set_trace()
im = caffe.io.load_image(caffe_root + 'examples/images/cat.jpg')
transformer = caffe.io.Transformer({'data': net_full_conv.blobs['data'].data.shape})
transformer.set_mean('data', np.load(caffe_root + 'python/caffe/imagenet/ilsvrc_2012_mean.npy').mean(1).mean(1))
transformer.set_transpose('data', (2,0,1))
transformer.set_channel_swap('data', (2,1,0))
transformer.set_raw_scale('data', 255.0)
# make classification map by forward and print prediction indices at each location
out = net_full_conv.forward_all(data=np.asarray([transformer.preprocess('data', im)]))
print out['prob'][0].argmax(axis=0)
