import numpy as np
import PIL.Image
import scipy.io as sio
import sys
import caffe
from caffe import caffe_utils as utils
import datetime

CAFFE_ROOT='/works/caffe/'

MODEL_GOOGLE_DEPLOY_FILE = '%s/models/bvlc_googlenet/deploy.prototxt' % CAFFE_ROOT
MODEL_GOOGLE_WEIGHT_FILE = '%s/models/bvlc_googlenet/bvlc_googlenet.caffemodel' % CAFFE_ROOT

MODEL_ORIGINAL_INPUT_SIZE = 256, 256
MODEL_INPUT_SIZE = 224, 224
MODEL_MEAN_VALUE = np.float32([104.0, 116.0, 122.0]) # bvlc_googlenet

DATASET_ROOT = '/storage/holidays/'
DATASET_LIST = 'holidays_db.txt'

FEATURE_JITTER = 10
FEATURE_DIM_GOOGLE = 1024
FEATURE_DIM_VGG = 4096

if __name__ == '__main__':
  #import pdb; pdb.set_trace()
  caffe.set_mode_cpu()
  net_google= caffe.Classifier( MODEL_GOOGLE_DEPLOY_FILE, MODEL_GOOGLE_WEIGHT_FILE, mean = MODEL_MEAN_VALUE, channel_swap = (2, 1, 0) ) 

  src_google = net_google.blobs['data']
  src_google.reshape(FEATURE_JITTER, 3, MODEL_INPUT_SIZE[0], MODEL_INPUT_SIZE[1])

  filenames=['%s' % entry.strip().split(' ')[0] for entry in open('%s/%s' % (DATASET_ROOT, DATASET_LIST))]
  feat_google= np.squeeze(np.zeros((len(filenames), FEATURE_JITTER, FEATURE_DIM_GOOGLE), dtype=np.float32))
  #import pdb; pdb.set_trace()

  for n, fname in enumerate(filenames):
    try:
      im = utils.load_image( '%s/jpg/%s' %(DATASET_ROOT, fname) )
      im = im.resize( MODEL_ORIGINAL_INPUT_SIZE, PIL.Image.ANTIALIAS )
      im = utils.preprocess(net_google, im)
    except:
      print 'error: filename: ', fname
    if FEATURE_JITTER == 10:
      im_jittered = utils.oversample(im, MODEL_INPUT_SIZE)
      src_google.data[:] = im_jittered
    else: src_google.data[:] = im[:]

    net_google.forward(end='pool5/7x7_s1')
    feat_google[n] = np.squeeze(net_google.blobs['pool5/7x7_s1'].data)

    if (n+1) % 10 == 0: print 'End of ', n+1; sys.stdout.flush()

  # save mat format
  #sio.savemat('%s_google_fc6_pool5_7x7_s1.mat' % DATASET_LIST, {'filenames': filenames, 'feat_vgg': feat_vgg, 'feat_google': feat_google})
  sio.savemat('%s_google_pool5_7x7_s1.mat' % DATASET_LIST, {'filenames': filenames, 'feat_google': feat_google})
