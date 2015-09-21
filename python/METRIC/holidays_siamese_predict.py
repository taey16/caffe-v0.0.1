import numpy as np
import PIL.Image
import scipy.io as sio
import sys
import caffe
from caffe import caffe_utils as utils
import datetime

CAFFE_ROOT='/works/caffe/'

MODEL_VGG_DEPLOY_FILE = '/storage/CDVS_Dataset/model/vgg/prototxt/deploy.prototxt'
MODEL_VGG_WEIGHT_FILE = '/storage/CDVS_Dataset/model/vgg/cdvs_vgg_simese_train_val_iter_64000.caffemodel'

MODEL_ORIGINAL_INPUT_SIZE = 384, 384
MODEL_INPUT_SIZE = 224, 224
MODEL_MEAN_VALUE = np.float32([104., 117., 123.]) # vgg-16

DATASET_ROOT = '/storage/holidays/'
DATASET_LIST = 'holidays_images.dat'

FEATURE_JITTER = 10
FEATURE_DIM_VGG = 128

if __name__ == '__main__':
  #import pdb; pdb.set_trace()
  caffe.set_mode_cpu()
  net_vgg   = caffe.Classifier( MODEL_VGG_DEPLOY_FILE, MODEL_VGG_WEIGHT_FILE, mean = MODEL_MEAN_VALUE, channel_swap = (2, 1, 0) ) 

  src_vgg = net_vgg.blobs['data']
  src_vgg.reshape(FEATURE_JITTER, 3, MODEL_INPUT_SIZE[0], MODEL_INPUT_SIZE[1])

  filenames=['%s' % entry.strip().split(' ')[0] for entry in open('%s/%s' % (DATASET_ROOT, DATASET_LIST))]
  feat_vgg   = np.squeeze(np.zeros((len(filenames), FEATURE_JITTER, FEATURE_DIM_VGG), dtype=np.float32))
  #import pdb; pdb.set_trace()

  for n, fname in enumerate(filenames):
    im = utils.load_image( '%s/jpg/%s' %(DATASET_ROOT, fname) )
    im = im.resize( MODEL_ORIGINAL_INPUT_SIZE, PIL.Image.ANTIALIAS )
    im = utils.preprocess(net_vgg, im)
    im_jittered = utils.oversample(im, MODEL_INPUT_SIZE)
    src_vgg.data[:] = im_jittered
    #import pdb; pdb.set_trace()
    feat = net_vgg.forward()
    feat_vgg[n] = feat[net_vgg.outputs[1]]

    if (n+1) % 10 == 0: print 'End of ', n+1; sys.stdout.flush()

  # save mat format
  sio.savemat('%s_%dx%d_vgg_siamese_fc7_dim256_embedding.mat' % (DATASET_LIST, MODEL_ORIGINAL_INPUT_SIZE[0], MODEL_ORIGINAL_INPUT_SIZE[1]), {'filenames': filenames, 'feat_vgg': feat_vgg})
