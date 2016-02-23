import caffe
import numpy as np
import sys

import pdb;pdb.set_trace()
mean_file = 'ResNet_mean.binaryproto'
blob = caffe.proto.caffe_pb2.BlobProto()
data = open(mean_file,'rb').read()
blob.ParseFromString(data)
arr = np.array( caffe.io.blobproto_to_array(blob) )
out = arr[0]
np.save( sys.argv[2] , out )
