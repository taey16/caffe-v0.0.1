import numpy as np
import scipy.io as sio
import sys, time
from compute_ap import *
from pop_counter import *

DATASET_ROOT = '/storage/holidays/'
DATASET_INPUT_LIST = 'eval_holidays/holidays_images.dat'
DATASET_GT_LIST = 'eval_holidays/perfect_result.dat'

#INPUT_MAT_FILENAME = 'holidays_images.dat_512x512_vggoogle_fc6_pool5_7x7_s1.mat'
INPUT_MAT_FILENAME = 'holidays_images.dat_384x384_vggoogle_fc6_pool5_7x7_s1.mat'
#INPUT_MAT_FILENAME = 'holidays_images.dat_vggoogle_fc6_pool5_7x7_s1.mat'

if __name__ == '__main__':
  filenames = [entry.strip() \
    for entry in open('%s/%s' % (DATASET_ROOT, DATASET_INPUT_LIST))]
  gt = dict([[entry.strip().split(' ')[0], entry.strip().split(' ')[::2]] \
    for entry in open('%s/%s' % (DATASET_ROOT, DATASET_GT_LIST))])
  mat = sio.loadmat('%s/%s' % (DATASET_ROOT, INPUT_MAT_FILENAME))
  fea_vgg, fea_google, filename = \
    mat['feat_vgg'].astype(np.float32), mat['feat_google'].astype(np.float32), mat['filenames']
  dic_ref, dic_idx = {}, {}
  for n, fname in enumerate(filename):
    dic_ref[n], dic_idx[fname] = fname, n

  #import pdb; pdb.set_trace()
  #fea_vgg = fea_vgg - np.mean(np.reshape(fea_vgg, (1491*10, 4096)), axis=0)
  #fea_google= fea_google - np.mean(np.reshape(fea_google, (1491*10, 1024)), axis=0)

  fea_vgg   =np.reshape(fea_vgg,   (fea_vgg.shape[0],fea_vgg.shape[1]*fea_vgg.shape[2]))
  fea_google=np.reshape(fea_google,(fea_google.shape[0],fea_google.shape[1]*fea_google.shape[2]))

  #import pdb; pdb.set_trace()
  sum_ap, num_query = 0., 0.
  fea = np.uint16(np.hstack((np.packbits(np.uint8(fea_vgg   >0), axis=1), 
                             np.packbits(np.uint8(fea_google>0), axis=1))))
  fea_shift = fea << 8
  fea = fea_shift[:,0::2] + fea[:,1::2]

  for query_fname, gt_result in gt.iteritems():
    query_id = dic_idx[query_fname]
    global_start = time.time()
    #start = time.time()
    diff = np.bitwise_xor(fea[query_id], fea)
    #print "bitwise_xor time: %.2gs" % (time.time() - start)
    #start = time.time()
    diff = lookup[diff]
    #print "lookup time: %.2gs" % (time.time() - start)
    #start = time.time()
    dist = np.sum(diff, axis=1)
    #print "sum over time: %.2gs" % (time.time() - start)

    #start = time.time()
    results = np.argsort(dist)
    #print "sort time: %.2gs" % (time.time() - start)
    results = results[1:]
    print "global: %.2gs" % (time.time() - global_start)

    print_str = 'End of %s\n' % query_fname
    for r, nn in enumerate(results):
      if r > 10: break
      print_str = print_str + '%s||%.4f ' %  (dic_ref[nn], dist[nn])
    print print_str

    tp_ranks = []
    for n, img_idx in enumerate(results):
      ref_fname = dic_ref[img_idx]
      if ref_fname in gt_result:
        tp_ranks.append(n)

    sum_ap += score_ap_from_ranks_1(tp_ranks, len(gt_result)-1); num_query += 1.
    tp_ranks_str = ''
    for rrr in tp_ranks:
      tp_ranks_str += str(rrr) + ' '
    print tp_ranks_str
    print 'mAP: ', sum_ap / num_query; sys.stdout.flush()

  print 'mAP: %.5f' % ( sum_ap / len(gt) )
