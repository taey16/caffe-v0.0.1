
import sys,pdb


def usage():
  print >> sys.stderr,"""usage: python holidays_map.py resultfile.dat

Where resultfile.dat is a textfile. Its format is:

result_file = ( result_line newline )*

# each line is a query image with associated results
result_line = query_image_name query_result*

# a query result is a pair: the result's filename is prefixed with its rank (0 based)
query_result = rank result_image_name 

Where:
- all items are separated by whitespaces (space or tab)
- image names are like 12345.jpg (case sensitive)
- the order of queries is not relevant
- if the query image is ranked, it is ignored in the scoring

Copyright INRIA 2008. License: GPL
"""
  sys.exit(1)

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

  
def get_groundtruth():
  """ Read datafile holidays_images.dat and output a dictionary
  mapping queries to the set of positive results (plus a list of all
  images)"""
  gt={}
  allnames=set()
  for line in open("/storage/ukbench/ukbench_gt.txt","r"):
    imname=line.strip()
    query = line.strip().split(' ')[0]
    gt[query]= set(line.strip().split(' ')[0::2])
    allnames.add(query)
  return (allnames,gt)

def print_perfect():
  " make a perfect result file "
  (allnames,gt)=get_groundtruth()
  for qname,results in gt.iteritems():
    print qname,
    for rank,resname in enumerate(results):
      print rank,resname,
    print

def parse_results(fname):
  """ go through the results file and return them in suitable
  structures"""
  for l in open(fname,"r"):
    fields=l.split()
    query_name=fields[0]
    ranks=[int(rank) for rank in fields[1::2]]
    yield (query_name,zip(ranks,fields[2::2]))


#########################################################################
# main program

if len(sys.argv)!=2: usage()

infilename=sys.argv[1]

import pdb; pdb.set_trace()
(allnames,gt)=get_groundtruth()

# sum of average precisions
#recall_4 = 0.
sum_ap=0.
# nb of images so far
n=0

# loop over result lines
for query_name,results in parse_results(infilename):
  
  if query_name not in gt:
    print "unknown query ",query_name
    sys.exit(1)

  # sort results by increasing rank
  results.sort()
  # ground truth
  gt_results=gt.pop(query_name)
  # ranks of true positives (not including the query)
  tp_ranks = []
  
  for rank,returned_name in results:
    if returned_name not in allnames:
      print "image name %s not in Holidays"%returned_name
      sys.exit(1)
    if returned_name in gt_results:
      tp_ranks.append(rank)      
  sum_ap+=score_ap_from_ranks_1(tp_ranks,len(gt_results))

  """
  hit = 0
  if tp_ranks[0] == 0: 
    hit = 0.25
  if tp_ranks[1] == 1: 
    hit = 0.5
  if tp_ranks[2] == 2: 
    hit = 0.75
  if tp_ranks[3] == 3: 
    hit = 1.0
  recall_4 += hit
  """

  n+=1

if gt:
  # some queries left
  print "no result for queries",gt.keys()
  sys.exit(1)

#print "recall@4 for %s: %.5f, ks: %f"%(infilename, recall_4/n, recall_4*4/n)
print "mAP for %s: %.5f"%(infilename, sum_ap/n)

