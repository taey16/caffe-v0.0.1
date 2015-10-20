import numpy as np
import PIL.Image
import scipy.io as sio
import cPickle as pickle
import sys
import caffe
from caffe import caffe_utils as utils
import datetime

CAFFE_ROOT='/works/caffe/'

MODEL_GOOGLE_DEPLOY_FILE = '%s/models/bvlc_googlenet/deploy.prototxt' % CAFFE_ROOT
MODEL_GOOGLE_WEIGHT_FILE = '%s/models/bvlc_googlenet/bvlc_googlenet.caffemodel' % CAFFE_ROOT
MODEL_VGG_DEPLOY_FILE = '%s/models/vgg/vgg_layer16_deploy.prototxt' % CAFFE_ROOT
MODEL_VGG_WEIGHT_FILE = '%s/models/vgg/vgg_layer16.caffemodel' % CAFFE_ROOT

MODEL_ORIGINAL_INPUT_SIZE = 384, 384
MODEL_INPUT_SIZE = 224, 224
MODEL_MEAN_VALUE = np.float32([103.939, 116.779, 123.68]) # vgg-16

DATASET_ROOT = '/storage/product/det/csv/'

FEATURE_JITTER = 10
FEATURE_DIM_GOOGLE = 1024
FEATURE_DIM_VGG = 4096
BIT_PACK_DIM = 3200
NUM_SAMPLE = 230000

if __name__ == '__main__':
  #import pdb; pdb.set_trace()
  caffe.set_mode_gpu()
  net_google= caffe.Classifier( \
    MODEL_GOOGLE_DEPLOY_FILE, MODEL_GOOGLE_WEIGHT_FILE, mean = MODEL_MEAN_VALUE, \
    channel_swap = (2, 1, 0)) 
  net_vgg = caffe.Classifier( \
    MODEL_VGG_DEPLOY_FILE, MODEL_VGG_WEIGHT_FILE, mean = MODEL_MEAN_VALUE, \
    channel_swap = (2, 1, 0)) 

  src_google= net_google.blobs['data']
  src_vgg   = net_vgg.blobs['data']
  src_google.reshape(FEATURE_JITTER, 3, MODEL_INPUT_SIZE[0], MODEL_INPUT_SIZE[1])
  src_vgg.reshape(   FEATURE_JITTER, 3, MODEL_INPUT_SIZE[0], MODEL_INPUT_SIZE[1])

  #import pdb; pdb.set_trace()
  for nn in range(1):
    INPUT_FILENAME = 'unique-labeller_eng_20150625144012.csv.cate_bbox.csv.shuffle_%02d.csv.readable_only.csv' % (nn)
    OUTPUT_FILENAME = '%s/%s.bit.pickle' % (DATASET_ROOT, INPUT_FILENAME)
    print 'Start ', INPUT_FILENAME

    filenames= [entry.strip().split('||')[2] \
      for entry in open('%s/%s' %( DATASET_ROOT, INPUT_FILENAME))];
    ref_database = np.zeros((NUM_SAMPLE, BIT_PACK_DIM), dtype=np.uint16)

    database={}
    dic_ref, dic_idx = {}, {}
    for n, fname in enumerate(filenames):
      if n >= NUM_SAMPLE: break
      fname_ = str(fname).strip().split('/')[-2:]
      fname__ = "/".join(fname_)
      dic_ref[n], dic_idx[fname__] = fname__, n

    #import pdb; pdb.set_trace()
    if len(dic_ref) <> len(dic_idx):
      print 'len(dic_ref) len(dic_idx) mismatched'
      sys.exit(-1)

    database['dic_ref'] = dic_ref
    database['dic_idx'] = dic_idx

    #import pdb; pdb.set_trace()
    for n, fname in enumerate(filenames):
      if n >= NUM_SAMPLE: break
      try:
        im = utils.load_image( '%s' %(fname) )
        im = im.resize( MODEL_ORIGINAL_INPUT_SIZE, PIL.Image.ANTIALIAS )
        im = utils.preprocess(net_google, im)
      except:
        print 'ERROR: filename: ', fname

      im_jittered = utils.oversample(im, MODEL_INPUT_SIZE)
      src_google.data[:], src_vgg.data[:] = im_jittered, im_jittered

      net_google.forward(end='pool5/7x7_s1')
      net_vgg.forward(end='fc6')
      feat_vgg = np.reshape(net_vgg.blobs['fc6'].data, (1,10*4096))
      feat_google = np.reshape(np.squeeze(net_google.blobs['pool5/7x7_s1'].data), (1,10*1024))
      feat = np.hstack((feat_vgg,feat_google))
      fea = (np.packbits(np.uint8(feat > 0), axis=1)).astype(np.uint16)
      fea_shift = fea << 8
      ref_database[n] = fea_shift[:,0::2] + fea[:,1::2]

      if (n+1) % 10 == 0: print 'End of ', n+1; sys.stdout.flush()

    database['ref'] = ref_database
    print 'Save to ', OUTPUT_FILENAME
    pickle.dump(database, open(OUTPUT_FILENAME, 'wb'))
