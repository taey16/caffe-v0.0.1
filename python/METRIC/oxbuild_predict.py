import numpy as np
import PIL.Image
import scipy.io as sio
import sys, datetime


CAFFE_ROOT='/works/caffe/'
sys.path.insert(0, CAFFE_ROOT + 'python')
import caffe;
from caffe import caffe_utils as utils

MODEL_ROOT='/storage/models/'

MODEL_GOOGLE_DEPLOY_FILE = '%s/bvlc_googlenet/deploy_feature_pool5_7x7_s1.prototxt' % MODEL_ROOT
MODEL_GOOGLE_WEIGHT_FILE = '%s/bvlc_googlenet/bvlc_googlenet.caffemodel' % MODEL_ROOT
MODEL_VGG_DEPLOY_FILE = '%s/vgg/vgg_layer16_deploy_feature_fc6.prototxt' % MODEL_ROOT
MODEL_VGG_WEIGHT_FILE = '%s/vgg/vgg_layer16.caffemodel' % MODEL_ROOT

MODEL_ORIGINAL_INPUT_SIZE = 384, 384
MODEL_INPUT_SIZE = 224, 224
#MODEL_MEAN_VALUE = np.float32([104.0, 116.0, 122.0]) # bvlc_googlenet
MODEL_MEAN_VALUE = np.float32([103.939, 116.779, 123.68]) # vgg-16

DATASET_ROOT = '/storage/oxbuild/'
DATASET_LIST = 'oxbuild_images.txt'

FEATURE_JITTER = 10
FEATURE_DIM_GOOGLE = 1024
FEATURE_DIM_VGG = 4096

if __name__ == '__main__':
  #import pdb; pdb.set_trace()
  caffe.set_mode_cpu()
  net_google= caffe.Classifier( MODEL_GOOGLE_DEPLOY_FILE, MODEL_GOOGLE_WEIGHT_FILE, mean = MODEL_MEAN_VALUE, channel_swap = (2, 1, 0) ) 
  net_vgg   = caffe.Classifier( MODEL_VGG_DEPLOY_FILE, MODEL_VGG_WEIGHT_FILE, mean = MODEL_MEAN_VALUE, channel_swap = (2, 1, 0) ) 

  src_google = net_google.blobs['data']
  src_google.reshape(FEATURE_JITTER, 3, MODEL_INPUT_SIZE[0], MODEL_INPUT_SIZE[1])
  src_vgg = net_vgg.blobs['data']
  src_vgg.reshape(FEATURE_JITTER, 3, MODEL_INPUT_SIZE[0], MODEL_INPUT_SIZE[1])

  filenames=['%s' % entry.strip() for entry in open('%s/%s' % (DATASET_ROOT, DATASET_LIST))]
  feat_vgg   = np.squeeze(np.zeros((len(filenames), FEATURE_JITTER, FEATURE_DIM_VGG), dtype=np.float32))
  feat_google= np.squeeze(np.zeros((len(filenames), FEATURE_JITTER, FEATURE_DIM_GOOGLE), dtype=np.float32))
  #import pdb; pdb.set_trace()

  for n, fname in enumerate(filenames):
    try:
      im = utils.load_image( '%s/oxbuild_images/%s' %(DATASET_ROOT, fname) )
      im = im.resize( MODEL_ORIGINAL_INPUT_SIZE, PIL.Image.ANTIALIAS )
      im = utils.preprocess(net_google, im)
    except:
      print 'error: filename: ', fname
    if FEATURE_JITTER == 10:
      im_jittered = utils.oversample(im, MODEL_INPUT_SIZE)
      src_google.data[:], src_vgg.data[:] = im_jittered, im_jittered
    else: src_google.data[:], src_vgg.data[:] = im[:], im[:]

    net_google.forward(end='pool5/7x7_s1')
    net_vgg.forward(end='fc6')
    feat_vgg[n] = net_vgg.blobs['fc6'].data
    feat_google[n] = np.squeeze(net_google.blobs['pool5/7x7_s1'].data)

    if (n+1) % 1 == 0: print 'End of ', n+1; sys.stdout.flush()

  # save mat format
  sio.savemat('%s_vggoogle_fc6_pool5_7x7_s1.mat' % DATASET_LIST, {'filenames': filenames, 'feat_vgg': feat_vgg, 'feat_google': feat_google})
