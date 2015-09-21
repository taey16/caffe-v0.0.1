import numpy as np
import scipy.io as sio
import sys

DATASET_ROOT = '/storage/holidays/'
DATASET_INPUT_LIST = 'eval_holidays/holidays_images.dat'
DATASET_GT_LIST = 'eval_holidays/perfect_result.dat'

#INPUT_MAT_FILENAME = 'holidays_images.dat_384x384_vggoogle_fc6_pool5_7x7_s1.mat'
#INPUT_MAT_FILENAME = 'holidays_images.dat_vggoogle_fc6_pool5_7x7_s1.mat'
INPUT_MAT_FILENAME = 'holidays_images.dat_384x384_vgg_siamese_fc7_dim128_embedding.mat'

FEATURE_JITTER = 10

def score_ap_from_ranks_1 (ranks, nres):
  """ Compute the average precision of one search.
  ranks = ordered list of ranks of true positives
  nres  = total number of positives in dataset  
  """
  # accumulate trapezoids in PR-plot
  ap=0.0
  # All have an x-size of:
  recall_step=1.0/nres
  for ntp,rank in enumerate(ranks):
    # y-size on left side of trapezoid:
    # ntp = nb of true positives so far
    # rank = nb of retrieved items so far
    if rank==0: precision_0=1.0
    else:       precision_0=ntp/float(rank)
    # y-size on right side of trapezoid:
    # ntp and rank are increased by one
    precision_1=(ntp+1)/float(rank+1)
    ap+=(precision_1+precision_0)*recall_step/2.0
  return ap


if __name__ == '__main__':
  filenames = [entry.strip() for entry in open('%s/%s' % (DATASET_ROOT, DATASET_INPUT_LIST))]
  gt = dict([[entry.strip().split(' ')[0], entry.strip().split(' ')[::2]] for entry in open('%s/%s' % (DATASET_ROOT, DATASET_GT_LIST))])
  mat = sio.loadmat(INPUT_MAT_FILENAME)
  vgg, filename = mat['feat_vgg'].astype(np.float32), mat['filenames']
  #fea_vgg = np.zeros_like(vgg)
  fea_vgg = vgg
  import pdb; pdb.set_trace()

  dic_ref, dic_idx = {}, {}
  for n, fname in enumerate(filename):
    dic_ref[n], dic_idx[fname] = fname, n

  sum_ap, num_query = 0., 0.
  for query_fname, gt_result in gt.iteritems():
    query_id = dic_idx[query_fname]
    diff_vgg = fea_vgg[query_id] - fea_vgg
    if FEATURE_JITTER == 1:
      # L2 dist.
      dist = np.sqrt(np.sum(diff**2, axis=1))
      # L1 dist.
      #dist = np.sum(np.abs(diff), axis=1)
    else:
      dist = np.mean(np.sqrt(np.sum(diff_vgg ** 2, axis=2)), axis=1)

    results = np.argsort(dist)
    results = results[1:]

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
