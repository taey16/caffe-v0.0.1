import numpy as np
import PIL.Image
import scipy.io as sio
import sys, os
import caffe
from caffe import caffe_utils as utils
import lmdb
import datetime
import random

CAFFE_ROOT='/works/caffe/'

DATASET_ROOT = '/storage/CDVS_Dataset/'
SET_ID = 'val'
DATASET_LIST = 'CDVS_all_l2_rank150_pairs.txt.shuffle.txt.%s.txt' % SET_ID
LMDB_PATH='./'
LMDB_FILENAME= 'CDVS1M_256_uint8_siamese_%s_lmdb' % SET_ID

MODEL_ORIGINAL_INPUT_SIZE = 256, 256

if __name__ == '__main__':
  #import pdb; pdb.set_trace()
  entries=[entry.strip().split(' ') for entry in open('%s/PAIRS/CDVS_1M_PAIRS/%s' % (DATASET_ROOT, DATASET_LIST))]
  #import pdb; pdb.set_trace()
  print 'Load data from: ', DATASET_LIST

  LMDB_NAME = '%s/%s' % (LMDB_PATH, LMDB_FILENAME)
  os.system('rm -rf %s' % (LMDB_NAME))
  env = lmdb.open('%s' %(LMDB_NAME), map_size=1e12)

  #import pdb; pdb.set_trace()
  n = 0
  for entry in entries:
    try:
      docid_i, docid_j, dist, label = entry[0], entry[1], entry[2], entry[3]
      im_i = utils.load_image( '%s/%s' %(DATASET_ROOT, docid_i))
      im_i = im_i.resize( MODEL_ORIGINAL_INPUT_SIZE, PIL.Image.ANTIALIAS )
      im_i = np.uint8(im_i)
      im_i = np.rollaxis(im_i, 2)[::-1]
      im_j = utils.load_image( '%s/%s' %(DATASET_ROOT, docid_j))
      im_j = im_j.resize( MODEL_ORIGINAL_INPUT_SIZE, PIL.Image.ANTIALIAS )
      im_j = np.uint8(im_j)
      im_j = np.rollaxis(im_j, 2)[::-1]
      im = np.vstack((im_i, im_j))
      datum = caffe.proto.caffe_pb2.Datum()
      datum.channels = im.shape[0]
      datum.height = im.shape[1]
      datum.width = im.shape[2]
      datum.data = im.tobytes()
      datum.label = int(label)
      str_id = '{:0>10d}'.format(n)

      with env.begin(write=True) as txn:
        txn.put(str_id, datum.SerializeToString())

      n+=1
    except:
      print 'ERROR ', docid_i, docid_j; sys.stdout.flush()

    if (n) % 100 == 0: print 'End of ', n; sys.stdout.flush()

  print 'END of creating lmdb: ', LMDB_NAME

  """
  datum = caffe.proto.caffe_pb2.Datum()
  datum.ParseFromString(value)
  label = int(datum.label)
  image = caffe.io.datum_to_array(datum)
  image = image.astype(np.uint8)

  import pdb; pdb.set_trace()
  env = lmdb.open('mylmdb', readonly=True)

  with env.begin() as txn:
    cur = txn.cursor()
    for k, v in cur:
      datum = caffe.proto.caffe_pb2.Datum()
      datum.ParseFromString(v)
      flat_img = np.fromstring(datum.data, dtype=np.uint8)
      img = flat_img.reshape(datum.channels, datum.height, datum.width)
      label = datum.label
      print k, datum.channels, datum.height, datum.width, label
  """
