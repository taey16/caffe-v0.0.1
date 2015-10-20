import numpy as np
import PIL.Image
#import sys
#import caffe
#from caffe import caffe_utils as utils
CAFFE_ROOT='/works/caffe/'
import sys; sys.path.insert(0, CAFFE_ROOT + 'python')
import caffe; caffe.set_mode_gpu()
from caffe import caffe_utils as utils

MODEL_VGG_DEPLOY_FILE = '%s/models/vgg/vgg_layer16_deploy_fcn.prototxt' % CAFFE_ROOT
MODEL_VGG_WEIGHT_FILE = '%s/models/vgg/vgg_layer16_fcn.caffemodel' % CAFFE_ROOT

MODEL_ORIGINAL_INPUT_SIZE = 1024, 1024
MODEL_INPUT_SIZE = 224, 224
MODEL_OUTPUT_SIZE= (1024-MODEL_INPUT_SIZE[0]) / 2**5 + 1
MODEL_MEAN_VALUE = np.float32([103.939, 116.779, 123.68]) # vgg

DATASET_ROOT = '/storage/CDVS_Dataset/'
#DATASET_LIST = 'database_images.txt.00.txt'
DATASET_LIST = '2_retrieval.txt'
#DATASET_LIST = '3_retrieval.txt'
#DATASET_LIST = '4_retrieval.txt'
#DATASET_LIST = '5_retrieval.txt'
#DATASET_LIST = '1a_retrieval.txt'
#DATASET_LIST = '1c_retrieval.txt'
#DATASET_LIST = '1b_retrieval.txt'

FEATURE_JITTER = MODEL_OUTPUT_SIZE**2
FEATURE_DIM_VGG = 4096


if __name__ == '__main__':
  import pdb; pdb.set_trace()
  net_vgg   = caffe.Classifier( MODEL_VGG_DEPLOY_FILE, MODEL_VGG_WEIGHT_FILE, mean = MODEL_MEAN_VALUE, channel_swap = (2, 1, 0) ) 

  src_vgg = net_vgg.blobs['data']
  src_vgg.reshape(1, 3, MODEL_ORIGINAL_INPUT_SIZE[0], MODEL_ORIGINAL_INPUT_SIZE[1])

  filenames=[entry.strip().split(' ') for entry in open('%s/%s' % (DATASET_ROOT, DATASET_LIST))]

  import pdb; pdb.set_trace()
  for n, fname in enumerate(filenames):
    query = fname[0]
    im = utils.load_image( '%s/%s' %(DATASET_ROOT, query) )
    im = im.resize( MODEL_ORIGINAL_INPUT_SIZE, PIL.Image.ANTIALIAS )
    im = utils.preprocess(net_vgg, im)
    src_vgg.data[:] = im[:]
    net_vgg.forward(end='fc6_conv')
    query_feat = np.copy(np.reshape(net_vgg.blobs['fc6_conv'].data, (1,FEATURE_DIM_VGG, MODEL_OUTPUT_SIZE**2))[0].T)
    #q_fea = np.argmax(np.abs(query_feat), axis=0)
    #q_fea = np.sign(query_feat) * np.max(np.abs(query_feat), axis=0)
    for n, f in enumerate(query_feat):
      query_feat[n] = f / np.linalg.norm(f,1)

    ref = filenames[10][1]
    im = utils.load_image( '%s/%s' %(DATASET_ROOT, ref) )
    im = im.resize( MODEL_ORIGINAL_INPUT_SIZE, PIL.Image.ANTIALIAS )
    im = utils.preprocess(net_vgg, im)
    src_vgg.data[:] = im[:]
    net_vgg.forward(end='fc6_conv')
    ref_feat = np.copy(np.reshape(net_vgg.blobs['fc6_conv'].data, (1,FEATURE_DIM_VGG, MODEL_OUTPUT_SIZE**2))[0].T)
    #ref_fea = np.argmax(np.abs(ref_feat), axis=0)
    #ref_fea = np.sign(ref_feat) * np.max(np.abs(ref_feat), axis=0)
    for n, f in enumerate(ref_feat):
      ref_feat[n] = f / np.linalg.norm(f,1)

    dist = 0.
    for f in query_feat:
      dist += np.sum(np.abs(f - ref_feat).ravel())

    if (n+1) % 10 == 0: print 'End of ', n+1; sys.stdout.flush()
