import numpy as np
import PIL.Image
import scipy.io as sio
import sys
import caffe
from caffe import caffe_utils as utils
import datetime

CAFFE_ROOT='/works/caffe/'

MODEL_DEPLOY_FILE = '/storage/ImageNet/ILSVRC2012/model/resnet/ResNet_caffe_models/ResNet-152-deploy.prototxt'
MODEL_WEIGHT_FILE = '/storage/ImageNet/ILSVRC2012/model/resnet/ResNet_caffe_models/ResNet-152-model.caffemodel'

MODEL_ORIGINAL_INPUT_SIZE = 256, 256
MODEL_INPUT_SIZE = 224, 224
MODEL_MEAN_FILE = '/storage/ImageNet/ILSVRC2012/model/resnet/ResNet_caffe_models/ResNet_mean.binaryproto'

DATASET_ROOT = '/storage/ImageNet/ILSVRC2012/val/'
DATASET_INPUT_LIST = '/storage/ImageNet/ILSVRC2012/val_synset.txt'

FEATURE_JITTER = 10


if __name__ == '__main__':
  import pdb; pdb.set_trace()
  caffe.set_device(1)
  caffe.set_mode_gpu()
  net = caffe.Classifier( \
    MODEL_GOOGLE_DEPLOY_FILE, 
    MODEL_GOOGLE_WEIGHT_FILE, 
    mean = MODEL_MEAN_VALUE, 
    channel_swap = (2, 1, 0) ) 

  layer_name = ['prob']

  input_blobs = net.blobs['data']
  input_blobs.reshape(\
    FEATURE_JITTER, 3, MODEL_INPUT_SIZE[0], MODEL_INPUT_SIZE[1])

  filenames=['%s/%s' % (DATASET_ROOT, entry.strip().split(' ')[0]) \
    for entry in open('%s' % DATASET_INPUT_LIST)]
  labels = [entry.strip().split(' ')[1] \
    for entry in open('%s' % DATASET_INPUT_LIST)]

  hit_count, hit5_count, tic_global = 0, 0, datetime.datetime.now()
  for n, fname in enumerate(filenames):
    try:
      tic_load = datetime.datetime.now()
      im = utils.load_image( fname )
      toc_load = datetime.datetime.now(); elapsed_load = toc_load - tic_load
      tic_resize = datetime.datetime.now()
      im = im.resize( MODEL_ORIGINAL_INPUT_SIZE, PIL.Image.ANTIALIAS )
      toc_resize = datetime.datetime.now(); elapsed_resize = toc_resize - tic_resize
      im = utils.preprocess(net, im)
      tic_jittering = datetime.datetime.now()
    except:
      print 'error: filename: ', fname
    if FEATURE_JITTER == 10:
      im_jittered = utils.oversample(im, MODEL_INPUT_SIZE)
      input_blobs.data[:] = im_jittered
    else: input_blobs.data[:] = im[:]
    toc_jittering = datetime.datetime.now(); 
    elapsed_jittering = toc_jittering - tic_jittering

    tic = datetime.datetime.now()
    net.forward(end=layer_name[0])
    toc = datetime.datetime.now(); elapsed = toc-tic
    dst = net.blobs[layer_name[0]].data

    score = np.mean(dst, axis=0)
    lab_pred=np.argsort(score)[::-1]

    if lab_pred[0] == int(labels[n]): hit_count += 1;
    if (lab_pred[0:5] == int(labels[n])).sum() == 1: hit5_count += 1

    if (n+1) % 100 == 0: 
      print 'acc@1: %f(%d/%d) acc@5: %f(%d/%d) in pred:%02.4f, load:%02.4f, resize:%02.4f, jitter:%02.4f msec.' % \
        ((hit_count/(n+1.0))*100.0, hit_count, n+1, (hit5_count/(n+1.0))*100, hit5_count, n+1.0, elapsed.microseconds/1000, elapsed_load.microseconds/1000, elapsed_resize.microseconds/1000, elapsed_jittering.microseconds/1000); sys.stdout.flush()

toc_global = datetime.datetime.now()
elapsed = toc_global - tic_global
print 'elapsed: %2.4f' % (elapsed.microseconds/1000)
