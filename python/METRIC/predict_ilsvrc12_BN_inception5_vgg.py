import numpy as np
import PIL.Image
import scipy.io as sio
import cPickle as pickle
import sys, traceback
import caffe
from caffe import caffe_utils as utils
import datetime

BN_INCEPTION_MODEL_ROOT='/storage/ImageNet/ILSVRC2012/model/bvlc_googlenet/'
VGG_MODEL_ROOT = '/storage/ImageNet/ILSVRC2012/model/vgg/'

MODEL_GOOGLE_DEPLOY_FILE = '%s/prototxt/deploy.prototxt' % BN_INCEPTION_MODEL_ROOT
MODEL_GOOGLE_WEIGHT_FILE = '%s/model/bvlc_googlenet.caffemodel' % BN_INCEPTION_MODEL_ROOT
MODEL_VGG_DEPLOY_FILE = '%s/prototxt/vgg_layer16_deploy_fcn.prototxt' % VGG_MODEL_ROOT
MODEL_VGG_WEIGHT_FILE = '%s/model/vgg_layer16_fcn.caffemodel' % VGG_MODEL_ROOT

MODEL_ORIGINAL_INPUT_SIZE = 256, 256
MODEL_INPUT_SIZE = 224, 224
MODEL_MEAN_VALUE = np.float32([103.939, 116.779, 123.68]) # vgg-16

DATASET_ROOT = '/storage/ImageNet/ILSVRC2012/'
DATASET_INPUT_LIST = '%s/val_synset.txt' % DATASET_ROOT

FEATURE_JITTER = 10
FEATURE_DIM_GOOGLE = 1024
FEATURE_DIM_VGG = 4096

layer_name = ['prob']

if __name__ == '__main__':
  #import pdb; pdb.set_trace()
  caffe.set_mode_gpu()
  net_google= caffe.Classifier( MODEL_GOOGLE_DEPLOY_FILE, MODEL_GOOGLE_WEIGHT_FILE, mean = MODEL_MEAN_VALUE, channel_swap = (2, 1, 0) ) 
  net_vgg   = caffe.Classifier( MODEL_VGG_DEPLOY_FILE, MODEL_VGG_WEIGHT_FILE, mean = MODEL_MEAN_VALUE, channel_swap = (2, 1, 0) ) 

  src_google = net_google.blobs['data']
  src_google.reshape(FEATURE_JITTER, 3, MODEL_INPUT_SIZE[0], MODEL_INPUT_SIZE[1])
  src_vgg = net_vgg.blobs['data']
  src_vgg.reshape(FEATURE_JITTER, 3, MODEL_INPUT_SIZE[0], MODEL_INPUT_SIZE[1])

  filenames=['%s/val/%s' % (DATASET_ROOT, entry.strip().split(' ')[0]) for entry in open('%s' % DATASET_INPUT_LIST)]
  labels = [entry.strip().split(' ')[1] for entry in open('%s' % DATASET_INPUT_LIST)]

  hit_count, hit5_count, tic_global = 0, 0, datetime.datetime.now()
  for n, fname in enumerate(filenames):
    try:
      tic_load = datetime.datetime.now()
      im = utils.load_image( fname )
      toc_load = datetime.datetime.now(); elapsed_load = toc_load - tic_load
      tic_resize = datetime.datetime.now()
      im = im.resize( MODEL_ORIGINAL_INPUT_SIZE, PIL.Image.ANTIALIAS )
      toc_resize = datetime.datetime.now(); elapsed_resize = toc_resize - tic_resize
      im = utils.preprocess(net_google, im)
      tic_jittering = datetime.datetime.now()
    except:
      print 'error: filename: ', fname
    if FEATURE_JITTER == 10:
      im_jittered = utils.oversample(im, MODEL_INPUT_SIZE)
      src_google.data[:] = im_jittered
      src_vgg.data[:] = im_jittered
    else: src_google.data[0], src_vgg.data[0] = im[:], im[:]
    toc_jittering = datetime.datetime.now(); elapsed_jittering = toc_jittering - tic_jittering

    tic = datetime.datetime.now()
    net_google.forward(end=layer_name[0])
    net_vgg.forward(end=layer_name[0])
    toc = datetime.datetime.now(); elapsed = toc-tic
    preds = np.vstack((net_google.blobs[layer_name[0]].data,np.squeeze(net_vgg.blobs[layer_name[0]].data)))
    #preds = np.squeeze(net_google.blobs[layer_name[0]].data)

    score = np.mean(preds, axis=0)
    lab_pred=np.argsort(score)[::-1]

    if lab_pred[0] == int(labels[n]): hit_count += 1;
    if (lab_pred[0:5] == int(labels[n])).sum() == 1: hit5_count += 1

    if (n+1) % 10 == 0: print 'acc@1: %f(%d/%d) acc@5: %f(%d/%d) in pred:%02.4f, load:%02.4f, resize:%02.4f, jitter:%02.4f msec.' % ((hit_count/(n+1.0))*100.0, hit_count, n+1, (hit5_count/(n+1.0))*100, hit5_count, n+1.0, elapsed.microseconds/1000, elapsed_load.microseconds/1000, elapsed_resize.microseconds/1000, elapsed_jittering.microseconds/1000); sys.stdout.flush()

toc_global = datetime.datetime.now()
elapsed = toc_global - tic_global
print 'elapsed: %2.4f' % (elapsed.microseconds/1000)
