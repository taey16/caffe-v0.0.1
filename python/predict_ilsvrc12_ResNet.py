import numpy as np
import sys
import caffe
from caffe import caffe_utils as utils
import datetime

CAFFE_ROOT='/works/caffe/'

MODEL_DEPLOY_FILE = \
  '/storage/ImageNet/ILSVRC2012/model/resnet/ResNet_caffe_models/ResNet-50-deploy.prototxt'
  #'/storage/ImageNet/ILSVRC2012/model/resnet/ResNet_caffe_models/ResNet-101-deploy.prototxt'
  #'/storage/ImageNet/ILSVRC2012/model/resnet/ResNet_caffe_models/ResNet-152-deploy.prototxt'
MODEL_WEIGHT_FILE = \
  '/storage/ImageNet/ILSVRC2012/model/resnet/ResNet_caffe_models/ResNet-50-model.caffemodel'
  #'/storage/ImageNet/ILSVRC2012/model/resnet/ResNet_caffe_models/ResNet-101-model.caffemodel'
  #'/storage/ImageNet/ILSVRC2012/model/resnet/ResNet_caffe_models/ResNet-152-model.caffemodel'

MODEL_ORIGINAL_INPUT_SIZE = 256, 256
MODEL_INPUT_SIZE = 224, 224
MODEL_MEAN_FILE = '/storage/ImageNet/ILSVRC2012/model/resnet/ResNet_caffe_models/ResNet_mean.binaryproto'
blob = caffe.proto.caffe_pb2.BlobProto()
data = open(MODEL_MEAN_FILE, 'rb').read()
blob.ParseFromString(data)
MODEL_MEAN_VALUE = np.squeeze(np.array( caffe.io.blobproto_to_array(blob) ))

DATASET_ROOT = '/storage/ImageNet/ILSVRC2012/val/'
DATASET_INPUT_LIST = '/storage/ImageNet/ILSVRC2012/val_synset.txt'

oversample = True

if __name__ == '__main__':
  #import pdb; pdb.set_trace()
  caffe.set_device(0)
  caffe.set_mode_gpu()
  net = caffe.Classifier( \
    model_file=MODEL_DEPLOY_FILE, 
    pretrained_file=MODEL_WEIGHT_FILE, 
    image_dims=(MODEL_ORIGINAL_INPUT_SIZE[0], MODEL_ORIGINAL_INPUT_SIZE[1]),
    raw_scale=255., # scale befor mean subtraction
    input_scale=None, # scale after mean subtraction
    mean = MODEL_MEAN_VALUE, 
    channel_swap = (2, 1, 0) ) 

  filenames=['%s/%s' % (DATASET_ROOT, entry.strip().split(' ')[0]) \
    for entry in open('%s' % DATASET_INPUT_LIST)]
  labels = [entry.strip().split(' ')[1] \
    for entry in open('%s' % DATASET_INPUT_LIST)]

  hit_count, hit5_count, tic_global = 0, 0, datetime.datetime.now()
  for n, fname in enumerate(filenames):
    try:
      #import pdb; pdb.set_trace()
      tic_load = datetime.datetime.now()
      im = caffe.io.load_image(fname)
      toc_load = datetime.datetime.now(); elapsed_load = toc_load - tic_load
      tic_resize = datetime.datetime.now()
      toc_resize = datetime.datetime.now(); elapsed_resize = toc_resize - tic_resize
      tic_predict = datetime.datetime.now()
      scores = net.predict([im], oversample) 
      elapsed = datetime.datetime.now() - tic_predict
    except Exception as err:
      print 'error: filename: ', fname
      print err

    score = np.mean(scores, axis=0)
    lab_pred=np.argsort(score)[::-1]

    if lab_pred[0] == int(labels[n]): hit_count += 1;
    if (lab_pred[0:5] == int(labels[n])).sum() == 1: hit5_count += 1

    if (n+1) % 1 == 0: 
      print 'acc@1: %f(%d/%d) acc@5: %f(%d/%d) in pred:%02.4f, load:%02.4f, resize:%02.4f msec.' % \
        ((hit_count/(n+1.0))*100.0, 
         hit_count, n+1, 
        (hit5_count/(n+1.0))*100, 
         hit5_count, n+1.0, 
        elapsed.microseconds/1000, 
        elapsed_load.microseconds/1000, 
        elapsed_resize.microseconds/1000); sys.stdout.flush()

toc_global = datetime.datetime.now()
elapsed = toc_global - tic_global
print 'elapsed: %2.4f' % (elapsed.microseconds/1000)
