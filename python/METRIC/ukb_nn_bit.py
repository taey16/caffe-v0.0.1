import numpy as np
import scipy.io as sio
import sys, time

DATASET_ROOT = '/storage/ukbench/'
DATASET_INPUT_LIST = 'ukbench_image.txt'
DATASET_GT_LIST = 'ukbench_gt.txt'

INPUT_MAT_FILENAME = 'ukbench_image.txt_384x384_vggoogle_fc6_pool5_7x7_s1.mat'


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
    else: precision_0=ntp/float(rank)
    # y-size on right side of trapezoid:
    # ntp and rank are increased by one
    precision_1=(ntp+1)/float(rank+1)
    ap+=(precision_1+precision_0)*recall_step/2.0
  return ap


if __name__ == '__main__':
  #import pdb; pdb.set_trace()
  filenames = [entry.strip() for entry in open('%s/%s' % (DATASET_ROOT, DATASET_INPUT_LIST))]
  gt = dict([[entry.strip().split(' ')[0], entry.strip().split(' ')[::2]] for entry in open('%s/%s' % (DATASET_ROOT, DATASET_GT_LIST))])
  mat = sio.loadmat('%s/%s' % (DATASET_ROOT, INPUT_MAT_FILENAME))
  fea_vgg, fea_google, filename = mat['feat_vgg'].astype(np.float32), mat['feat_google'].astype(np.float32), mat['filenames']

  dic_ref, dic_idx = {}, {}
  for n, fname in enumerate(filename):
    dic_ref[n] = fname; dic_idx[fname] = n

  sum_ap, num_query = 0., 0.
  #fea_vgg, fea_google = fea_vgg > 0, fea_google > 0 
  fea_vgg, fea_google = np.packbits(np.uint8(fea_vgg > 0), axis=2), np.packbits(np.uint8(fea_google > 0), axis=2)
  for query_fname, gt_result in gt.iteritems():
    query_id = dic_idx[query_fname]
    start = time.time()
    diff_vgg, diff_google = np.bitwise_xor(fea_vgg[query_id], fea_vgg), np.bitwise_xor(fea_google[query_id], fea_google) 
    dist_vgg = np.sum(diff_vgg , axis=2)
    dist_google = np.sum(diff_google, axis=2)
    dist = np.sum(np.hstack((dist_vgg, dist_google)), axis=1)
    print "hd time: %.2gs" % (time.time() - start)

    start = time.time()
    results = np.argsort(dist)
    print "sort time: %.2gs" % (time.time() - start)

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
