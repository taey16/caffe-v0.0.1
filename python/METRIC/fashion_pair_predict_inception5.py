import sys
import os
import numpy as np
from numpy import linalg as LA
import PIL.Image
def add_path(path):
  if path not in sys.path:
    sys.path.insert(0, path)
caffe_path = os.path.join('/works/caffe', 'python')
hamming_dist_path = os.path.join('/works/demon_11st/indexer','hamming_distance')
add_path(caffe_path)
add_path(hamming_dist_path)
import caffe
from caffe import caffe_utils as utils
import hamming_distance as h_dist


MODEL_GOOGLE_DEPLOY_FILE = \
  '/storage/ImageNet/ILSVRC2012/model/inception5/inception5_deploy.prototxt'
MODEL_GOOGLE_WEIGHT_FILE = \
  '/storage/ImageNet/ILSVRC2012/model/inception5/inception5_deploy.caffemodel'

MODEL_ORIGINAL_INPUT_SIZE = 256, 256
MODEL_INPUT_SIZE = 224, 224
MODEL_MEAN_VALUE = np.float32([104.0, 116.0, 122.0])

FEATURE_JITTER = 10
#FEATURE_JITTER = 1

input_filename = '/storage/product/fashion_pair/fashion_pair_test.csv'
prefix = '/data1/october_11st/october_11st_imgs/'
output_filename = '/works/caffe-v0.0.1/python/METRIC/fashion_pair_predict_inception5.py'+'.jitter'+str(FEATURE_JITTER)+'.hd'+'.log'

entries = \
  [entry.strip().split(',') for entry in open(input_filename, 'r')]
log = open(output_filename,'w')
if __name__ == '__main__':
  #import pdb; pdb.set_trace()
  caffe.set_device(0)
  caffe.set_mode_gpu()
  net = caffe.Classifier( 
    MODEL_GOOGLE_DEPLOY_FILE, MODEL_GOOGLE_WEIGHT_FILE, 
    mean = MODEL_MEAN_VALUE, channel_swap = (2, 1, 0) ) 

  src = net.blobs['data']
  src.reshape(FEATURE_JITTER, 3, MODEL_INPUT_SIZE[0], MODEL_INPUT_SIZE[1])

  for n, fname in enumerate(entries):
    try:
      query_filename = fname[0] 
      ref_filename = fname[1]
      label = int(fname[2])
      inputs_q  = utils.load_image( os.path.join(prefix, query_filename) )
      inputs_ref= utils.load_image( os.path.join(prefix, ref_filename) )

      if FEATURE_JITTER == 10:
        inputs_q  = inputs_q.resize( MODEL_ORIGINAL_INPUT_SIZE, PIL.Image.ANTIALIAS )
        inputs_ref= inputs_ref.resize( MODEL_ORIGINAL_INPUT_SIZE, PIL.Image.ANTIALIAS )
        inputs_q  = utils.preprocess(net, inputs_q)
        inputs_ref= utils.preprocess(net, inputs_ref)
        inputs_q  = utils.oversample(inputs_q, MODEL_INPUT_SIZE)
        inputs_ref= utils.oversample(inputs_ref, MODEL_INPUT_SIZE)
      else:
        inputs_q  = inputs_q.resize( MODEL_INPUT_SIZE, PIL.Image.ANTIALIAS )
        inputs_ref= inputs_ref.resize( MODEL_INPUT_SIZE, PIL.Image.ANTIALIAS )
        inputs_q  = utils.preprocess(net, inputs_q)
        inputs_ref= utils.preprocess(net, inputs_ref)
        inputs_q  = inputs_q[np.newaxis,:,:,:]
        inputs_ref= inputs_ref[np.newaxis,:,:,:]

      src.data[:] = inputs_q
      net.forward(end='pool5/7x7_s1')
      feat_q  = np.squeeze(net.blobs['pool5/7x7_s1'].data).copy()
      src.data[:] = inputs_ref
      net.forward(end='pool5/7x7_s1')
      feat_ref= np.squeeze(net.blobs['pool5/7x7_s1'].data)

      #if FEATURE_JITTER == 10:
      #  feat_q  = np.mean(feat_q,axis=0)
      #  feat_ref= np.mean(feat_ref,axis=0)
      
      #import pdb; pdb.set_trace()
      #feat_q = feat_q / LA.norm(feat_q, 2.0)
      #feat_ref = feat_ref / LA.norm(feat_ref, 2.0)
      if FEATURE_JITTER == 10:
        feat_q = np.reshape(feat_q, (1,feat_q.shape[0]*feat_q.shape[1]))
        feat_ref = np.reshape(feat_ref,(1,feat_ref.shape[0]*feat_ref.shape[1]))
      else:
        feat_q = np.reshape(feat_q, (1,feat_q.shape[0]))
        feat_ref = np.reshape(feat_ref,(1,feat_ref.shape[0]))
      bins = np.array([0],dtype=np.uint8)
      dig_feat_q = np.digitize(feat_q,bins,right=True)
      dig_feat_ref = np.digitize(feat_ref,bins,right=True)
      feat_q = np.uint64(np.packbits(np.uint8(dig_feat_q),axis=1))
      feat_ref = np.uint64(np.packbits(np.uint8(dig_feat_ref),axis=1))
      import pdb; pdb.set_trace()
      for i in range (0,3):
        q_shift = feat_q << 8*(2**i) 
        ref_shift = feat_ref << 8*(2**i)
        feat_q = q_shift[:,0::2] + feat_q[:,1::2]
        feat_ref = ref_shift[:,0::2] + feat_ref[:,1::2]
  
      distance = h_dist.hamming_distance(feat_q,feat_ref)
      #distance = LA.norm(feat_q - feat_ref, 2.0)
      #distance = np.sum(np.sqrt(np.power(feat_q - feat_ref, 2.0)))
      print('%d %s %s %d %d' %(n, query_filename, ref_filename, distance, label))
      log.write('%d\t%d\n' %(label, distance))
    except:
      print 'ERROR: query_filename: ', fname[0]


log.close()
