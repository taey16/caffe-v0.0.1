import os, sys, time, datetime
import cPickle
import logging
import optparse
import pandas as pd
import numpy as np
from PIL import Image
import cStringIO as StringIO
import urllib
import exifutil
import PIL.Image

# caffe import
CAFFE_ROOT = '/works/caffe/'
import sys
sys.path.insert(0, CAFFE_ROOT + 'python')
import caffe;
from caffe import caffe_utils as caffe_utils

#DATABASE_FILENAME = '/storage/product/det/unique-labeller_eng_20150625144012.csv.cate_bbox.csv.shuffle_00.csv.readable_only.csv.bit.pickle.webpath.pickle.inception5.pickle'
#DATABASE_FILENAME = '/storage/product/11st_6M/11st_380K.shuffle.webpath.bit.pickle.inception5.pickle'
DATABASE_FILENAME = '/storage/product/11st_6M/11st_380K.shuffle.webpath.bit.pickle.inception5.4096bit.pickle'
NUM_NEIGHBORS = 10
UPLOAD_FOLDER = '/tmp/caffe_demos_uploads'
ENROLL_FOLDER = '/storage/enroll/'
ALLOWED_IMAGE_EXTENSIONS = set(['png', 'bmp', 'jpg', 'jpe', 'jpeg', 'gif'])


def classify_url(clf, imageurl):
  try:
    string_buffer = StringIO.StringIO(urllib.urlopen(imageurl).read())
    image = caffe.io.load_image(string_buffer)

  except Exception as err:
    # For any exception we encounter in reading the image, we will just
    # not continue.
    logging.info('URL Image open error: %s', err)
    return False, None

  logging.info('Image: %s', imageurl)
  result = clf.classify_image(image)
  return True, result


def classify_upload(imagefile):
  try:
    # We will save the file to disk for possible data collection.
    filename_ = str(datetime.datetime.now()).replace(' ', '_') + \
      werkzeug.secure_filename(imagefile.filename)
    filename = os.path.join(UPLOAD_FOLDER, filename_)
    imagefile.save(filename)
    logging.info('Saving to %s.', filename)
    image = exifutil.open_oriented_im(filename)

  except Exception as err:
    logging.info('Uploaded image open error: %s', err)
    return flask.render_template(
      'index.html', has_result=True,
      result=(False, 'Cannot open uploaded image.')
    )

  result = app.clf.classify_image(image)
  return result


def embed_image_html(image):
  """Creates an image embedded in HTML base64 format."""
  image_pil = Image.fromarray((255 * image).astype('uint8'))
  image_pil = image_pil.resize((256, 256))
  string_buf = StringIO.StringIO()
  image_pil.save(string_buf, format='png')
  data = string_buf.getvalue().encode('base64').replace('\n', '')
  return 'data:image/png;base64,' + data


def allowed_file(filename):
  return (
    '.' in filename and
    filename.rsplit('.', 1)[1] in ALLOWED_IMAGE_EXTENSIONS
  )


class ImagenetClassifier(object):
  default_args = {
    'model_def_file': ( '/storage/models/bvlc_googlenet/deploy.prototxt'),
    'pretrained_model_file': (
      '/storage/models/bvlc_googlenet/bvlc_googlenet.caffemodel'),
    'class_labels_file': (
      '{}/data/ilsvrc12/synset_words.txt'.format(CAFFE_ROOT)),
  }
  for key, val in default_args.iteritems():
    if not os.path.exists(val):
      raise Exception(
        "File for {} is missing. Should be at: {}".format(key, val))

  default_args['image_dim'] = 384
  default_args['raw_scale'] = 255.

  # reference database
  database_param = '%s' % DATABASE_FILENAME

  def __init__(self, model_def_file, pretrained_model_file,
         raw_scale, class_labels_file, image_dim, gpu_mode):
    logging.info('Loading net and associated files...')
    if gpu_mode: caffe.set_mode_gpu()
    else: caffe.set_mode_cpu()

    ## load models googlenet
    self.net = caffe.Classifier(
      model_def_file, pretrained_model_file,
      image_dims=(image_dim, image_dim), raw_scale=raw_scale,
      mean=np.array([104.0, 116.0, 122.0]), channel_swap=(2, 1, 0))
    logging.info('Load vision model, %s', model_def_file)

    # generate N bit lookup table
    self.lookup = np.asarray([bin(i).count('1') for i in range(1<<16)])

    # load reference bit model
    file_reader = open(self.database_param, 'rb')
    self.database = cPickle.load(file_reader)
    file_reader.close()
    logging.info('Load database from {}'.format(self.database_param))
    logging.info('database shape {}'.format(self.database['ref'].shape))

    with open(class_labels_file) as f:
      labels_df = pd.DataFrame([
        {
          'synset_id': l.strip().split(' ')[0],
          'name': ' '.join(l.strip().split(' ')[1:]).split(',')[0]
        }
        for l in f.readlines()
      ])
    self.labels = labels_df.sort('synset_id')['name'].values


  def enroll_image(self, image, filename):
    try:
      # predict
      scores = self.net.predict([image], oversample=True).flatten()
      # extract features for retrieval
      logging.info('pool5/7x7_s1 shape: {}'.format(self.net.blobs['pool5/7x7_s1'].data.shape))
      feat = np.reshape(np.squeeze(self.net.blobs['pool5/7x7_s1'].data), (1,10*1024))
      logging.info('feat shape: {}'.format(feat.shape))
      # binalize and 16bit-bitpacking
      fea = (np.packbits(np.uint8(feat > 0), axis=1)).astype(np.uint16)
      fea_shift = fea << 8
      fea = fea_shift[:,0::2] + fea[:,1::2]
      enrolled_img_path = '/enroll/' + filename
      self.database['dic_ref'][self.database['ref'].shape[0]] = str(enrolled_img_path)
      self.database['dic_idx'][enrolled_img_path] = self.database['ref'].shape[0]
      self.database['ref'] = np.vstack((self.database['ref'], fea))
      logging.info("query shape: {}".format(fea.shape))
      logging.info('ref shape: {}'.format(self.database['ref'].shape))
      logging.info('Enroll done')
      return (False, 'Enroll complete')
    except Exception as err:
      logging.info('Enroll error: %s', err)
      return (False, \
        'Something went wrong when enrolling the ' 'image. Maybe try another one?')
  

  def classify_image(self, image):
    try:
      # inference
      global_starttime = time.time()
      scores = self.net.predict([image], oversample=False).flatten()
      # sort top-5 label
      indices = (-scores).argsort()[:5]
      predictions = self.labels[indices]
      endtime = time.time()
      logging.info('Predict done for %d classes in %f', scores.shape[0], endtime - global_starttime)
  
      # extract features for retrieval
      logging.info('pool5/7x7_s1 shape: {}'.format(self.net.blobs['pool5/7x7_s1'].data.shape))

      starttime = time.time()
      feat = np.reshape(np.squeeze(self.net.blobs['pool5/7x7_s1'].data), (1,10*1024))
      endtime = time.time()
      logging.info('feat shape: {}'.format(feat.shape))
      #logging.info('feat_vgg.norm: %s', str(np.linalg.norm(feat_vgg[0,:],1)))
      #logging.info('feat_google.norm: %s', str(np.linalg.norm(feat_google[0,:],1)))
      # binalize and 16bit-bitpacking
      fea = (np.hstack((np.packbits(np.uint8(feat > 0), axis=1)))).astype(np.uint16)
      fea_shift = fea << 8
      fea = fea_shift[0::2] + fea[1::2]
      logging.info("query shape: {}".format(fea.shape))
      logging.info('Hashing done in %f', endtime - starttime)

      # nearest-neighbor
      starttime = time.time()
      diff = np.bitwise_xor(self.database['ref'], fea)
      diff = self.lookup[diff]
      dist = np.sum(diff, axis=1)
      neighbor_list = np.argsort(dist)
      endtime = time.time()
      logging.info('Nearest Neighbor done in %f', endtime - starttime)
      logging.info("dist shape: {}".format(dist.shape))
      logging.info("neighbor shape: {}".format(neighbor_list.shape))
      result_neighbor = []
      for n, neighbor in enumerate(neighbor_list[0:NUM_NEIGHBORS]):
        logging.info("top-{}: {}, {}".format(n, self.database['dic_ref'][neighbor], dist[neighbor]))
        # general web path
        result_neighbor.append('10.202.211.120:2596/PBrain/%s' % (self.database['dic_ref'][neighbor]))

      # In addition to the prediction text, we will also produce
      # the length for the progress bar visualization.
      meta = [
        (p, '%.5f' % scores[i])
        for i, p in zip(indices, predictions)
      ]
      logging.info('result: %s', str(meta))

      return (True, meta, '%.3f' % (endtime - global_starttime), str(''), result_neighbor)

    except Exception as err:
      logging.info('Classification error: %s', err)
      return (False, 'Something went wrong when classifying the ' 'image. Maybe try another one?')


def start_from_terminal():
  """
  Parse command line options and start the server.
  """
  parser = optparse.OptionParser()
  parser.add_option(
    '-g', '--gpu',
    help="use gpu mode",
    action='store_true', default=True)

  opts, args = parser.parse_args()
  ImagenetClassifier.default_args.update({'gpu_mode': opts.gpu})

  # Initialize classifier + warm start by forward for allocation
  clf = ImagenetClassifier(**ImagenetClassifier.default_args)
  clf.net.forward()
  
  import pdb; pdb.set_trace()
  classify_url(clf, 'http://blog.trashness.com/wp-content/uploads/2012/12/boat-shoe-socks-men-fair-isle.jpg')


if __name__ == '__main__':
  import pdb; pdb.set_trace()
  logging.getLogger().setLevel(logging.INFO)
  if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)
  start_from_terminal()

