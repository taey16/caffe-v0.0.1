import numpy as np
import scipy.io as sio
import sys
from compute_ap import *

DATASET_ROOT = '/storage/ukbench/'
DATASET_INPUT_LIST = 'ukbench_image.txt'
DATASET_GT_LIST = 'ukbench_gt.txt'

INPUT_MAT_FILENAME = 'ukbench_image.txt_384x384_vggoogle_fc6_pool5_7x7_s1.mat'

FEATURE_JITTER = 10
FEATURE_NORM = 1


if __name__ == '__main__':
  #import pdb; pdb.set_trace()
  filenames = [entry.strip() \
              for entry in open('%s/%s' % (DATASET_ROOT, DATASET_INPUT_LIST))]
  gt = dict([[entry.strip().split(' ')[0], entry.strip().split(' ')[::2]] \
            for entry in open('%s/%s' % (DATASET_ROOT, DATASET_GT_LIST))])
  mat = sio.loadmat('%s/%s' % (DATASET_ROOT, INPUT_MAT_FILENAME))
  vgg = mat['feat_vgg'].astype(np.float32)
  google = mat['feat_google'].astype(np.float32)
  filename = mat['filenames']
  fea_vgg, fea_google = np.zeros_like(vgg), np.zeros_like(google)

  dic_ref, dic_idx = {}, {}
  for n, fname in enumerate(filename):
    dic_ref[n] = fname; dic_idx[fname] = n

  if FEATURE_JITTER == 1:
    for n, fea in enumerate(vgg): 
      fea_vgg[n] = fea / np.linalg.norm(fea, FEATURE_NORM)
    for n, fea in enumerate(google): 
      fea_google[n] = fea / np.linalg.norm(fea, FEATURE_NORM)
  else:
    for n, fea in enumerate(vgg): 
      fea_vgg[n] = (fea.T / np.linalg.norm(fea.T, FEATURE_NORM, axis=0)).T
    for n, fea in enumerate(google): 
      fea_google[n] = (fea.T / np.linalg.norm(fea.T, FEATURE_NORM, axis=0)).T

  sum_ap, num_query = 0., 0.
  for query_fname, gt_result in gt.iteritems():
    query_id = dic_idx[query_fname]
    diff_vgg, diff_google = fea_vgg[query_id] - fea_vgg, fea_google[query_id] - fea_google
    if FEATURE_JITTER == 1:
      # L2 dist.
      #dist = np.sqrt(np.sum(diff**2, axis=1))
      # L1 dist.
      dist = np.sum(np.abs(diff), axis=1)
    else:
      dist_vgg = np.sum(np.abs(diff_vgg), axis=2)
      dist_google = np.sum(np.abs(diff_google), axis=2)
      dist = np.mean(np.hstack((dist_vgg, dist_google)), axis=1)

    results = np.argsort(dist)

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
