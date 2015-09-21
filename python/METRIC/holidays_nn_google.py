import numpy as np
import scipy.io as sio
import sys

DATASET_ROOT = '/storage/holidays/'
DATASET_INPUT_LIST = 'eval_holidays/holidays_images.dat'
DATASET_GT_LIST = 'eval_holidays/perfect_result.dat'

FEATURE_JITTER = 10

if __name__ == '__main__':

  filenames = [entry.strip() for entry in open('%s/%s' % (DATASET_ROOT, DATASET_INPUT_LIST))]
  entries = dict([[entry.strip().split(' ')[0], entry.strip().split(' ')[::2]] for entry in open('%s/%s' % (DATASET_ROOT, DATASET_GT_LIST))])
  import pdb; pdb.set_trace()
  mat = sio.loadmat('holidays_images.dat_vggoogle_drop6_pool5_7x7_s1.mat')
  google, filename = mat['feat_vgg'].astype(np.float32), mat['filenames']
  fea_google = np.zeros_like(google)

  dic, dic_idx = {}, {}
  for n, fname in enumerate(filename):
    dic[n] = fname
    dic_idx[fname] = n

  if FEATURE_JITTER == 1:
    for n, fea in enumerate(google): fea_google[n] = fea / np.linalg.norm(fea, 1)
  else:
    for n, fea in enumerate(google): fea_google[n] = (fea.T / np.linalg.norm(fea.T, 1, axis=0)).T

  fo = open('holidays_google_drop6.mat.dat', 'w')
  for query_fname, gt_list in entries.iteritems():
    fo.write(query_fname + ' ')
    query_id = dic_idx[query_fname]
    diff_google = fea_google[query_id] - fea_google
    if FEATURE_JITTER == 1:
      # L2 dist.
      #dist = np.sqrt(np.sum(diff**2, axis=1))
      # L1 dist.
      dist = np.sum(np.abs(diff), axis=1)
    else:
      dist = np.sum(np.abs(diff_google), axis=2)
      dist = np.mean(dist, axis=1)

    rank = np.argsort(dist)

    rank_filename, rank_idx = [], 0
    for n, img_idx in enumerate(rank):
      ref_fname = dic[img_idx]
      try: hit = gt_list.index(ref_fname)
      except: hit = -1
      if hit > -1:
        rank_filename.append(str(rank_idx))
        rank_filename.append(ref_fname)
      rank_idx+=1
    fo.write(' '.join(rank_filename) + '\n'); sys.stdout.flush()
  fo.close()
