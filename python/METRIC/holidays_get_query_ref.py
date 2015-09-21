
import sys

def get_groundtruth():
  """ Read datafile holidays_images.dat and output a dictionary
  mapping queries to the set of positive results (plus a list of all
  images)"""
  gt={}
  allnames=set()
  for line in open("holidays_images.dat","r"):
    imname=line.strip()
    allnames.add(imname)
    imno=int(imname[:-len(".jpg")])    
    if imno%100==0:
      gt_results=set()
      gt[imname]=gt_results
    else:
      gt_results.add(imname)
      
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

import pdb; pdb.set_trace()
(allnames,gt)=get_groundtruth()

fo_q = open('holidays_query.txt', 'w')
fo_r = open('holidays_db.txt', 'w')
for k, v in gt.items():
  fo_q.write(k.strip() + '\n') 
  for item in v:
   fo_r.write(item.strip() + '\n')

fo_q.close()
fo_r.close()

print 'End'
