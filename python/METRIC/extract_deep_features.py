import numpy as np
import matplotlib.pyplot as plt
#%matplotlib inline
import scipy.io as sio
import PIL.Image
import sys

CAFFE_ROOT = '/Users/1002596/Documents/caffe/'
sys.path.insert(0, CAFFE_ROOT + '/python')
import caffe
from caffe import caffe_utils as utils

MODEL_DEPLOY_FILE = '%s/models/bvlc_googlenet/deploy.prototxt' % CAFFE_ROOT
MODEL_WEIGHT_FILE = '%s/models/bvlc_googlenet/bvlc_googlenet.caffemodel' % CAFFE_ROOT
#MODEL_DEPLOY_FILE = '%s/models/vgg/vgg_layer16_deploy_feature.prototxt' % CAFFE_ROOT
#MODEL_WEIGHT_FILE = '%s/models/vgg/vgg_layer16.caffemodel' % CAFFE_ROOT

MODEL_ORIGINAL_INPUT_SIZE = 224, 224 
MODEL_INPUT_SIZE = 224, 224
MODEL_MEAN_VALUE = np.float32([104.0, 116.0, 122.0]) # bvlc_googlenet
#MODEL_MEAN_VALUE = np.float32([103.939, 116.779, 123.68]) # vgg-16

#DATASET_ROOT = '/Users/1002596/Documents/neuraltalk/data/'
#DATASET_INPUT_LIST = 'holidays/eval_holidays/holidays_images.dat'
#DATASET_GT_LIST = 'holidays/eval_holidays/perfect_result.dat'
DATASET_ROOT = 'DATA/ukbench/'
DATASET_INPUT_LIST = '/ukbench_image.txt'
DATASET_GT_LIST = '/ukbench_gt.txt'

FEATURE_DIM = 4096 #25088 #4096 #1024
FEATURE_JITTER = 1

if __name__ == '__main__':

  # load net model
  caffe.set_mode_cpu()
  net = caffe.Classifier( MODEL_DEPLOY_FILE, MODEL_WEIGHT_FILE, mean = MODEL_MEAN_VALUE, channel_swap = (2, 1, 0) ) 
  layer_name = ['pool5', 'fc6', 'fc7', 'fc8', 'prob'] # vgg
  #layer_name = ['pool5/7x7_s1', 'prob'] # googlenet
  src = net.blobs['data']
  src.reshape(FEATURE_JITTER, 3, 224, 224)

  # load holidays image file list
  filenames=[entry.strip() for entry in open('%s/%s' % (DATASET_ROOT, DATASET_INPUT_LIST ))]
  # load gt list
  entries = [entry.strip().split(' ') for entry in open('%s/%s' % (DATASET_ROOT, DATASET_GT_LIST))]
  # set feature set
  feat, n = np.squeeze(np.zeros((len(filenames), FEATURE_JITTER, FEATURE_DIM), dtype=np.float32)), 0 
  import pdb; pdb.set_trace()

  for fname in filenames:
    # load img
    #im = PIL.Image.open('%s/holidays/jpg/%s' % (DATASET_ROOT, fname))
    im = PIL.Image.open('%s/full/%s' % (DATASET_ROOT, fname))
    im = im.resize(MODEL_ORIGINAL_INPUT_SIZE, PIL.Image.ANTIALIAS )
    # preprocessing
    im = utils.preprocess(net, im)
    if FEATURE_JITTER == 10:
      src.data[:] = utils.oversample(im, MODEL_INPUT_SIZE)
    else: src.data[:] = im[:]

    net.forward(end=layer_name[2])
    dst = net.blobs[layer_name[2]]
    feat[n] = dst.data.reshape(1, FEATURE_DIM)
    n+=1
    if n % 10 == 0: print 'End of %d' % n 

  # save mat format
  sio.savemat('ukb_vgg16_fc7.mat', {'feat': feat})
