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
#MODEL_DEPLOY_FILE = '%s/models/vgg/vgg_layer16_deploy_feature.prototxt' % CAFFE_ROOT
MODEL_VGG_DEPLOY_FILE = '%s/models/vgg/vgg_layer16_deploy.prototxt' % CAFFE_ROOT
MODEL_VGG_WEIGHT_FILE = '%s/models/vgg/vgg_layer16.caffemodel' % CAFFE_ROOT

MODEL_ORIGINAL_INPUT_SIZE = 256, 256
MODEL_INPUT_SIZE = 224, 224
#MODEL_MEAN_FILE = '%s/python/caffe/imagenet/ilsvrc_2012_mean.npy' % CAFFE_ROOT
MODEL_MEAN_VALUE = np.float32([104.0, 116.0, 122.0]) # bvlc_googlenet
#MODEL_MEAN_VALUE = np.float32([103.939, 116.779, 123.68]) # vgg-16

DATASET_ROOT = '/storage/ImageNet/ILSVRC2012/val/'
DATASET_INPUT_LIST = '%s/data/ilsvrc12/val.txt' % CAFFE_ROOT

FEATURE_JITTER = 10
FEATURE_DIM = 1000

if __name__ == '__main__':
  #import pdb; pdb.set_trace()
  caffe.set_mode_gpu()
  net_google= caffe.Classifier( MODEL_GOOGLE_DEPLOY_FILE, MODEL_GOOGLE_WEIGHT_FILE, mean = MODEL_MEAN_VALUE, channel_swap = (2, 1, 0) ) 
  net_vgg   = caffe.Classifier( MODEL_VGG_DEPLOY_FILE, MODEL_VGG_WEIGHT_FILE, mean = MODEL_MEAN_VALUE, channel_swap = (2, 1, 0) ) 

  #layer_name = ['pool5', 'fc6', 'fc7', 'fc8', 'prob'] # vgg
  layer_name = ['pool5/7x7_s1','','','','prob'] # googlenet

  src_google = net_google.blobs['data']
  src_google.reshape(FEATURE_JITTER, 3, MODEL_INPUT_SIZE[0], MODEL_INPUT_SIZE[1])
  src_vgg = net_vgg.blobs['data']
  src_vgg.reshape(FEATURE_JITTER, 3, MODEL_INPUT_SIZE[0], MODEL_INPUT_SIZE[1])

  filenames=['%s/%s' % (DATASET_ROOT, entry.strip().split(' ')[0]) for entry in open('%s' % DATASET_INPUT_LIST)]
  labels = [entry.strip().split(' ')[1] for entry in open('%s' % DATASET_INPUT_LIST)]
  #import pdb; pdb.set_trace()

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
      src_google.data[:], src_vgg.data[:] = im_jittered, im_jittered
    else: src_google.data[:], src_vgg.data[:] = im[:], im[:]
    toc_jittering = datetime.datetime.now(); elapsed_jittering = toc_jittering - tic_jittering

    tic = datetime.datetime.now()
    net_google.forward(end=layer_name[4])
    net_vgg.forward(end=layer_name[4])
    toc = datetime.datetime.now(); elapsed = toc-tic
    dst = np.vstack((net_google.blobs[layer_name[4]].data,net_vgg.blobs[layer_name[4]].data))

    score = np.mean(dst, axis=0)
    lab_pred=np.argsort(score)[::-1]

    if lab_pred[0] == int(labels[n]): hit_count += 1;
    if (lab_pred[0:5] == int(labels[n])).sum() == 1: hit5_count += 1

    if (n+1) % 100 == 0: print 'acc@1: %f(%d/%d) acc@5: %f(%d/%d) in pred:%02.4f, load:%02.4f, resize:%02.4f, jitter:%02.4f msec.' % ((hit_count/(n+1.0))*100.0, hit_count, n+1, (hit5_count/(n+1.0))*100, hit5_count, n+1.0, elapsed.microseconds/1000, elapsed_load.microseconds/1000, elapsed_resize.microseconds/1000, elapsed_jittering.microseconds/1000); sys.stdout.flush()

toc_global = datetime.datetime.now()
elapsed = toc_global - tic_global
print 'elapsed: %2.4f' % (elapsed.microseconds/1000)
