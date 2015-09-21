
import numpy as np
from scipy import io as sio
import cPickle

DATASET_ROOT = '/storage/CDVS_Dataset/'
#INPUT_FILENAME = 'ukbench_image.txt_384x384_vggoogle_fc6_pool5_7x7_s1.mat'
#INPUT_FILENAME = 'holidays_images.dat_384x384_vggoogle_fc6_pool5_7x7_s1.mat'
INPUT_FILENAME = 'database_images.txt_384x384_vggoogle_fc6_pool5_7x7_s1.mat'
import pdb;pdb.set_trace()
mat = sio.loadmat('%s/%s' % (DATASET_ROOT, INPUT_FILENAME))
fea_vgg, fea_google, filename = mat['feat_vgg'].astype(np.float32), mat['feat_google'].astype(np.float32), mat['filenames']
dic_ref, dic_idx = {}, {}
for n, fname in enumerate(filename):
  dic_ref[n] = str(fname.strip()); dic_idx[fname] = n

fea_vgg = np.reshape(fea_vgg, (fea_vgg.shape[0],fea_vgg.shape[1]*fea_vgg.shape[2]))
fea_google = np.reshape(fea_google, (fea_google.shape[0],fea_google.shape[1]*fea_google.shape[2]))
fea = (np.hstack((np.packbits(np.uint8(fea_vgg > 0), axis=1), np.packbits(np.uint8(fea_google > 0), axis=1)))).astype(np.uint16)
fea_shift = fea << 8
ref_database = fea_shift[:,0::2] + fea[:,1::2]
#ref = np.vstack((fea[:,0::2], np.zeros((1000000,fea.shape[1]/2)).astype(np.uint16)))
database={}
database['dic_ref'] = dic_ref
database['dic_idx'] = dic_idx
database['ref'] = ref_database

#sio.savemat('%s/%s.bit.mat' % (DATASET_ROOT, INPUT_FILENAME), database)
file_writer = open( '%s/%s.bit.pickle' % (DATASET_ROOT, INPUT_FILENAME), 'wb')
cPickle.dump(database, file_writer)
file_writer.close()

"""
import pdb;pdb.set_trace()
file_reader = open('%s/%s.bit.pickle' % (DATASET_ROOT, INPUT_FILENAME), 'rb')
cPickle.load(file_reader)
file_reader.close()
"""
