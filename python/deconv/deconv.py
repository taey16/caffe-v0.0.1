import numpy as np
import PIL.Image

CAFFE_ROOT = '/works/caffe/'
import sys
sys.path.insert(0, CAFFE_ROOT + 'python')
import caffe;
from caffe import caffe_utils as utils

MODEL_ROOT = '/storage/models/'
MODEL_VGG_DEPLOY_FILE = '%s/vgg/vgg_layer16_deploy_fcn_deconv.prototxt' % MODEL_ROOT
MODEL_VGG_WEIGHT_FILE = '%s/vgg/vgg_layer16_fcn.caffemodel' % MODEL_ROOT

MODEL_ORIGINAL_INPUT_SIZE = 224, 224
MODEL_MEAN_VALUE = np.float32([103.939, 116.779, 123.68]) # vgg-16

DATASET_ROOT = '/storage/CDVS_Dataset/'
DATASET_LIST = 'database_images.txt'

import pdb; pdb.set_trace()
caffe.set_mode_cpu()
net_vgg   = caffe.Classifier( MODEL_VGG_DEPLOY_FILE, MODEL_VGG_WEIGHT_FILE, mean = MODEL_MEAN_VALUE, channel_swap = (2, 1, 0) )
src_vgg = net_vgg.blobs['data']
src_vgg.reshape(1, 3, MODEL_ORIGINAL_INPUT_SIZE[0], MODEL_ORIGINAL_INPUT_SIZE[1])

filenames=['%s' % entry.strip().split(' ')[0] for entry in open('%s/%s' % (DATASET_ROOT, DATASET_LIST))]

end_blob_name  = ['pool1', 'pool2', 'pool3', 'pool4', 'pool5', 'fc6_conv', 'relu6_conv']
copy_to_blob_name= ['deconv2_1', 'deconv3_1', 'deconv4_1', 'deconv5_1', 'fc6_deconv', 'relu6_deconv', 'fc7_deconv']
decon_layer_name = ['unpool1', 'unpool2', 'unpool3', 'unpool4', 'unpool5', 'fc6_deconv', 'relu6_deconv']
iter_idx = [0, 1, 2, 3, 4, 5, 6]

print 'Start deconvolution'
for file_id, filename in enumerate(filenames):
  im = utils.load_image( '%s/%s' %(DATASET_ROOT, filename) )
  im = im.resize( MODEL_ORIGINAL_INPUT_SIZE, PIL.Image.ANTIALIAS )
  im = utils.preprocess(net_vgg, im); src_vgg.data[:] = im[:]
  print 'Done load image'

  for layer_idx, end_blob, copy_to, deconv_layer in zip(iter_idx, end_blob_name, copy_to_blob_name, decon_layer_name):
    net_vgg.forward(end=end_blob)
    net_vgg.blobs[copy_to].data[...] = np.copy(net_vgg.blobs[end_blob].data)
    net_vgg.forward(start=deconv_layer)
  
    recon = np.copy(net_vgg.blobs['deconv1_1'].data[0])
    recon = utils.deprocess(net_vgg, recon)
    min_val, max_val = recon.flatten().min(), recon.flatten().max()
    print "{}, layer {}, dim: {}={}, mean: {}, min_val: {}, max_val: {}".format( \
      filename, end_blob, net_vgg.blobs[end_blob].data[0].shape, \
      np.prod(net_vgg.blobs[end_blob].data[0].shape), \
      np.mean(recon.flatten()), min_val, max_val)
    recon = (recon - min_val) / (max_val - min_val)
    recon = recon * 255
    recon = np.uint8(np.clip(recon, 0, 255))
    recon_image = PIL.Image.fromarray(recon)
    image_filename = filename.split('/')[-1]
    recon_image.save('%s/deconv_images/%s_%s.png' % (DATASET_ROOT, image_filename, end_blob_name[layer_idx]))

    if (file_id+1) % 10 == 0: print 'Done {}/{}'.format(DATASET_ROOT, image_filename)

print 'Done deconvolution'
