import numpy as np
import cPickle as pickle
import sys, time
from compute_ap import *
from pop_counter import *

DATASET_ROOT = '/storage/product/det'
DATASET_INPUT_LIST = 'csv/unique-labeller_eng_20150625144012.csv.cate_bbox.csv.shuffle_00.csv.readable_only.csv'
#INPUT_MAT_FILENAME = 'unique-labeller_eng_20150625144012.csv.cate_bbox.csv.shuffle_00.csv.bit.pickle'
INPUT_MAT_FILENAME = 'unique-labeller_eng_20150625144012.csv.cate_bbox.csv.shuffle_00.csv.readable_only.csv.bit.pickle'

if __name__ == '__main__':
  filenames = [entry.strip() \
    for entry in open('%s/%s' % (DATASET_ROOT, DATASET_INPUT_LIST))]
  database = pickle.load(open('%s/%s' % (DATASET_ROOT, INPUT_MAT_FILENAME), 'rb'))

  #import pdb; pdb.set_trace()
  fea = database['ref']
  dic_idx = database['dic_idx']
  dic_ref = database['dic_ref']

  for query_id, query_fname in dic_ref.iteritems():
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
    #print "global: %.2gs" % (time.time() - global_start)

    print_str = 'End of %s\n' % query_fname
    for r, nn in enumerate(results):
      if r > 10: break
      print_str = print_str + '%s||%.4f ' %  (dic_ref[nn], dist[nn])
    print print_str
