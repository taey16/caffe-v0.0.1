import numpy as np
import scipy.io as sio
import sys, time
from compute_ap import *
from pop_counter import *


DATASET_ROOT = '/storage/ukbench/'
DATASET_INPUT_LIST = 'ukbench_image.txt'
DATASET_GT_LIST = 'ukbench_gt.txt'

INPUT_MAT_FILENAME = 'ukbench_image.txt_384x384_vggoogle_fc6_pool5_7x7_s1.mat'

if __name__ == '__main__':
  filenames = [entry.strip() \
    for entry in open('%s/%s' % (DATASET_ROOT, DATASET_INPUT_LIST))]
  gt = dict([[entry.strip().split(' ')[0], entry.strip().split(' ')[::2]] \
             for entry in open('%s/%s' % (DATASET_ROOT, DATASET_GT_LIST))])
  mat = sio.loadmat('%s/%s' % (DATASET_ROOT, INPUT_MAT_FILENAME))
  fea_vgg  = mat['feat_vgg'].astype(np.float32)
  fea_google = mat['feat_google'].astype(np.float32)
  filename = mat['filenames']

  dic_ref, dic_idx = {}, {}
  for n, fname in enumerate(filename):
    dic_ref[n] = fname; dic_idx[fname] = n

  fea_vgg   = np.reshape(fea_vgg,   (fea_vgg.shape[0],fea_vgg.shape[1]*fea_vgg.shape[2]))
  fea_google= np.reshape(fea_google,(fea_google.shape[0],fea_google.shape[1]*fea_google.shape[2]))
  import pdb; pdb.set_trace()

  # 16bit packing
  fea = (np.hstack((np.packbits(np.uint8(fea_vgg   > 0), axis=1), \
                    np.packbits(np.uint8(fea_google> 0), axis=1)))).astype(np.uint16)
  fea_shift = fea << 8
  fea = fea_shift[:,0::2] + fea[:,1::2]
  """
  # 32bit packing
  fea = (np.hstack((np.packbits(np.uint8(fea_vgg > 0), axis=1), np.packbits(np.uint8(fea_google > 0), axis=1)))).astype(np.uint32)
  fea_shift = fea << 8
  fea = fea_shift[:,0::2] + fea[:,1::2]
  fea_shift = fea << 16
  fea = fea_shift[:,0::2] + fea[:,1::2]
  """
  """
  # 64bit packing
  fea = (np.hstack((np.packbits(np.uint8(fea_vgg > 0), axis=1), np.packbits(np.uint8(fea_google > 0), axis=1)))).astype(np.uint64)
  fea_shift = fea << 8
  fea = fea_shift[:,0::2] + fea[:,1::2]
  fea_shift = fea << 16
  fea = fea_shift[:,0::2] + fea[:,1::2]
  fea_shift = fea << 32
  fea = fea_shift[:,0::2] + fea[:,1::2]
  """

  #fea = np.vstack((fea, np.zeros((220000,fea.shape[1])).astype(np.uint16)))

  sum_ap, num_query = 0., 0.
  for query_fname, gt_result in gt.iteritems():
    query_id = dic_idx[query_fname]
    global_time = time.time()
    start = time.time()
    diff = np.bitwise_xor(fea[query_id], fea)
    print "bitwise_xor time: %.2gs" % (time.time() - start)
    start = time.time()
    diff = lookup[diff]
    print "lookup time: %.2gs" % (time.time() - start)
    #diff = count_bits_u32(diff)
    #print "count_bits_u32 time: %.2gs" % (time.time() - start)
    #diff = bit_count_parallel_64(diff)
    #print "count_bits_u64 time: %.2gs" % (time.time() - start)
    start = time.time()
    dist = np.sum(diff, axis=1)
    print "sum over time: %.2gs" % (time.time() - start)

    start = time.time()
    results = np.argsort(dist)
    print "sort time: %.2gs" % (time.time() - start)
    print "global: %.2gs" %(time.time() - global_time)

    #import pdb; pdb.set_trace()
    print_str = 'End of %s\n' % query_fname
    for r, nn in enumerate(results):
      if r > 10: break
      print_str = print_str + '%s||%.4f ' %  (dic_ref[nn], dist[nn])
    print print_str

    #import pdb; pdb.set_trace()
    tp_ranks = []
    for n, img_idx in enumerate(results):
      ref_fname = dic_ref[img_idx]
      if ref_fname in gt_result[1:]: 
        tp_ranks.append(n)

    sum_ap += score_ap_from_ranks_1(tp_ranks, len(gt_result[1:]))
    num_query += 1.
    tp_ranks_str = ''
    for rrr in tp_ranks:
      tp_ranks_str += str(rrr) + ' ' 
    print tp_ranks_str
    print 'mAP: ', sum_ap / num_query; sys.stdout.flush()

  print 'mAP: %.5f' % ( sum_ap / len(gt) )
