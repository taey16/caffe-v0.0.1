import numpy as np
import scipy.io as sio
import sys
from compute_ap import *
from pop_counter import *


CATE_ID = '1a'

DATASET_ROOT = '/storage/CDVS_Dataset/'
DATASET_INPUT_LIST = '%s_retrieval.txt' % CATE_ID
DATASET_GT_LIST = '%s_retrieval.txt' % CATE_ID
MAT_Q_FILENAME = '%s/%s_retrieval.txt_384x384_vggoogle_fc6_pool5_7x7_s1.mat' % (DATASET_ROOT, CATE_ID)
MAT_R_FILENAME = '%s/database_images.txt_384x384_vggoogle_fc6_pool5_7x7_s1.mat' % (DATASET_ROOT)

FEATURE_JITTER = 10
FEATURE_NORM = 2


if __name__ == '__main__':

  #import pdb; pdb.set_trace()
  gt = dict([[entry.strip().split(' ')[0], entry.strip().split(' ')] \
            for entry in open('%s/%s' % (DATASET_ROOT, DATASET_GT_LIST), 'r')])
  mat = sio.loadmat(MAT_Q_FILENAME)
  query_vgg, query_google, filename_query = mat['feat_vgg'].astype(np.float32), mat['feat_google'].astype(np.float32), mat['filenames']
  mat = sio.loadmat(MAT_R_FILENAME)
  ref_vgg, ref_google, filename_ref = mat['feat_vgg'].astype(np.float32), mat['feat_google'].astype(np.float32), mat['filenames']
  #import pdb; pdb.set_trace()
  print 'Start nn,', MAT_Q_FILENAME, MAT_R_FILENAME

  dic_query, dic_query_idx = {}, {}
  for n, fname in enumerate(filename_query):
    fname = str(fname).strip()
    dic_query[n], dic_query_idx[fname] = fname, n
  dic_ref, dic_ref_idx = {}, {}
  for n, fname in enumerate(filename_ref):
    fname = str(fname).strip()
    dic_ref[n], dic_ref_idx[fname] = fname, n

  # bit-packing for ref
  fea_vgg = np.reshape(ref_vgg, (ref_vgg.shape[0],ref_vgg.shape[1]*ref_vgg.shape[2]))
  fea_google = np.reshape(ref_google, (ref_google.shape[0],ref_google.shape[1]*ref_google.shape[2]))
  fea = (np.hstack((np.packbits(np.uint8(fea_vgg > 0), axis=1), np.packbits(np.uint8(fea_google > 0), axis=1)))).astype(np.uint16)
  fea_shift = fea << 8
  ref = fea_shift[:,0::2] + fea[:,1::2]
  # bit-packing for query
  query_fea_vgg = np.reshape(query_vgg, (query_vgg.shape[0],query_vgg.shape[1]*query_vgg.shape[2]))
  query_fea_google = np.reshape(query_google, (query_google.shape[0],query_google.shape[1]*query_google.shape[2]))
  fea = (np.hstack((np.packbits(np.uint8(query_fea_vgg > 0), axis=1), np.packbits(np.uint8(query_fea_google > 0), axis=1)))).astype(np.uint16)
  fea_shift = fea << 8
  query = fea_shift[:,0::2] + fea[:,1::2]

  #import pdb; pdb.set_trace()
  sum_ap, num_query = 0., 0
  for query_fname, gt_result in gt.iteritems():
    query_id = dic_query_idx[query_fname]
    diff = np.bitwise_xor(query[query_id], ref)
    diff = lookup[diff]
    dist = np.sum(diff, axis=1)
    results = np.argsort(dist)

    #import pdb; pdb.set_trace()
    """
    print_str = 'End of %s\n' % query_fname
    for r, nn in enumerate(results):
      if r > 10: break
      print_str = print_str + '%s||%.4f ' %  (dic_ref[nn], dist[nn])
    print print_str
    """

    tp_ranks = []
    for n, img_idx in enumerate(results):
      ref_fname = dic_ref[img_idx]
      if ref_fname in gt_result[1:]: 
        tp_ranks.append(n)

    sum_ap += score_ap_from_ranks_1(tp_ranks, len(gt_result[1:]))
    num_query += 1.
    """
    tp_ranks_str = ''
    for rrr in tp_ranks:
      tp_ranks_str += str(rrr) + ' ' 
    print tp_ranks_str
    """
    print 'mAP: ', sum_ap / num_query; sys.stdout.flush()

  print 'mAP: %.5f' % ( sum_ap / len(gt) )
