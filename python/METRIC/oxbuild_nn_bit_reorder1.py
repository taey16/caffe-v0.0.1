import numpy as np
import scipy.io as sio
import sys
import compute_ap

#loop-up lookup for N bit
N = 16; lookup = np.asarray([bin(i).count('1') for i in range(1<<N)])

DATASET_ROOT = '/storage/oxbuild/'
DATASET_INPUT_LIST = 'oxbuild_images.txt'
DATASET_GT_LIST = 'oxbuild_gt.txt'

INPUT_MAT_FILENAME = 'oxbuild_images.txt_384x384_vggoogle_fc6_pool5_7x7_s1.mat'

"""
def score_ap_from_ranks_1 (ranks, nres):
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
"""

if __name__ == '__main__':

  import pdb; pdb.set_trace()
  gt = dict([[entry.strip().split(' ')[0], entry.strip().split(' ')[::2]] for entry in open('%s/%s' % (DATASET_ROOT, DATASET_GT_LIST))])
  mat = sio.loadmat('%s/%s' % (DATASET_ROOT, INPUT_MAT_FILENAME))
  fea_vgg, fea_google, filename = mat['feat_vgg'].astype(np.float32), mat['feat_google'].astype(np.float32), mat['filenames']
  dic_ref, dic_idx = {}, {}
  for n, fname in enumerate(filename):
    dic_ref[n] = str(fname.strip()); dic_idx[str(fname.strip())] = n

  fea_vgg = np.reshape(fea_vgg, (fea_vgg.shape[0],fea_vgg.shape[1]*fea_vgg.shape[2]))
  fea_google = np.reshape(fea_google, (fea_google.shape[0],fea_google.shape[1]*fea_google.shape[2]))

  fea = (np.hstack((np.packbits(np.uint8(fea_vgg > 0), axis=1), np.packbits(np.uint8(fea_google > 0), axis=1)))).astype(np.uint16)
  fea_shift = fea << 8
  fea = fea_shift[:,0::2] + fea[:,1::2]

  import pdb; pdb.set_trace()
  sum_ap, num_query = 0., 0.
  for query_fname, gt_result in gt.iteritems():
    query_id = dic_idx[query_fname]
    diff = np.bitwise_xor(fea[query_id], fea)
    diff = lookup[diff]
    dist = np.sum(diff, axis=1)

    results = np.argsort(dist)
    #results = results[1:]

    #import pdb; pdb.set_trace()
    print_str = 'End of %s\n' % query_fname
    for r, nn in enumerate(results[1:]):
      if r > 10: break
      print_str = print_str + '%s||%.4f ' %  (dic_ref[nn], dist[nn])
    print print_str

    #import pdb; pdb.set_trace()
    tp_ranks = []
    for n, img_idx in enumerate(results):
      ref_fname = dic_ref[img_idx]
      #if ref_fname in gt_result[1:]: 
      if ref_fname in gt_result: 
        tp_ranks.append(n)

    sum_ap += compute_ap.score_ap_from_ranks_1(tp_ranks, len(gt_result[1:]))
    num_query += 1.
    tp_ranks_str = ''
    for rrr in tp_ranks:
      tp_ranks_str += str(rrr) + ' ' 
    print tp_ranks_str
    print 'mAP: ', sum_ap / num_query; sys.stdout.flush()

  print 'mAP: %.5f' % ( sum_ap / len(gt) )