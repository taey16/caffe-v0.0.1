import os, sys
import time
import cPickle
import datetime
import logging
import flask
import werkzeug
import optparse
import tornado.wsgi
import tornado.httpserver
import numpy as np
import pandas as pd
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

# lstm import
RNN_ROOT = '/works/neuraltalk/'
sys.path.append(RNN_ROOT)
from imagernn.solver import Solver
from imagernn.imagernn_utils import decodeGenerator, eval_split


DATABASE_NAME = 'CDVS_Dataset'
DATABASE_ROOT = '/storage/%s/' % DATABASE_NAME
DATABASE_FILENAME = '%s/database_images.txt_384x384_vggoogle_fc6_pool5_7x7_s1.mat.bit.pickle' % DATABASE_ROOT
#DATABASE_FILENAME = '%s/holidays_images.dat_384x384_vggoogle_fc6_pool5_7x7_s1.mat.bit.pickle' % DATABASE_ROOT
#DATABASE_FILENAME = '%s/ukbench_image.txt_384x384_vggoogle_fc6_pool5_7x7_s1.mat.bit.pickle' % DATABASE_ROOT
NUM_NEIGHBOR = 10
REPO_DIRNAME = os.path.abspath(os.path.dirname(os.path.abspath(__file__)) + '/../..')
UPLOAD_FOLDER = '/tmp/caffe_demos_uploads'
ENROLL_FOLDER = '/storage/CDVS_Dataset/enroll/'
ALLOWED_IMAGE_EXTENSIONS = set(['png', 'bmp', 'jpg', 'jpe', 'jpeg', 'gif'])

# Obtain the flask app object
app = flask.Flask(__name__)


@app.route('/')
def index():
  return flask.render_template('index.html', has_result=False)


@app.route('/classify_url', methods=['GET'])
def classify_url():
  imageurl = flask.request.args.get('imageurl', '')
  try:
    string_buffer = StringIO.StringIO(urllib.urlopen(imageurl).read())
    image = caffe.io.load_image(string_buffer)

  except Exception as err:
    # For any exception we encounter in reading the image, we will just
    # not continue.
    logging.info('URL Image open error: %s', err)
    return flask.render_template(
      'index.html', has_result=True,
      result=(False, 'Cannot open image from URL.')
    )

  logging.info('Image: %s', imageurl)
  result = app.clf.classify_image(image)
  return flask.render_template(
    'index.html', has_result=True, result=result, imagesrc=imageurl)


@app.route('/enroll_upload', methods=['POST'])
def enroll_upload():
  try:
    # We will save the file to disk for possible data collection.
    imagefile = flask.request.files['imagefile_enroll']
    filename_ = str(datetime.datetime.now()).replace(' ', '_') + \
      werkzeug.secure_filename(imagefile.filename)
    filename = os.path.join(ENROLL_FOLDER, filename_)
    imagefile.save(filename)
    image = exifutil.open_oriented_im(filename)
    im = PIL.Image.fromarray(np.asarray(image * 255.).astype(np.uint8))
    im = im.resize( (256, 256), PIL.Image.ANTIALIAS )
    thumb_filename = filename + '_thumb.jpg'
    im.save(thumb_filename)
    scp_command = 'scp %s 1002596@10.202.211.120:/storage/CDVS_Dataset/enroll/' % thumb_filename
    os.system(scp_command)
    logging.info('Saving to %s. done', thumb_filename)
    logging.info('%s done', scp_command)

  except Exception as err:
    logging.info('Uploaded image open error: %s', err)
    return flask.render_template(
      'index.html', has_result=True,
      result=(False, 'Cannot open uploaded image.')
    )

  result = app.clf.enroll_image(image, filename_)
  return flask.render_template('index.html', has_result=True, result=result, imagesrc=embed_image_html(image))


@app.route('/classify_upload', methods=['POST'])
def classify_upload():
  try:
    # We will save the file to disk for possible data collection.
    imagefile = flask.request.files['imagefile']
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
  return flask.render_template('index.html', has_result=True, result=result, imagesrc=embed_image_html(image))


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
    'model_def_file': ( '{}/models/vgg/vgg_layer16_deploy.prototxt'.format(REPO_DIRNAME)),
    'pretrained_model_file': (
      '{}/models/vgg/vgg_layer16.caffemodel'.format(REPO_DIRNAME)),
    'mean_file': (
      '{}/python/caffe/imagenet/ilsvrc_2012_mean.npy'.format(REPO_DIRNAME)),
    'class_labels_file': (
      '{}/data/ilsvrc12/synset_words.txt'.format(REPO_DIRNAME)),
    'bet_file': (
      '{}/data/ilsvrc12/imagenet.bet.pickle'.format(REPO_DIRNAME)),
  }
  googlenet_args = {
    'model_def_file': ( '{}/models/bvlc_googlenet/deploy.prototxt'.format(REPO_DIRNAME)),
    'pretrained_model_file': (
      '{}/models/bvlc_googlenet/bvlc_googlenet.caffemodel'.format(REPO_DIRNAME)),
    'class_labels_file': (
      '{}/data/ilsvrc12/synset_words.txt'.format(REPO_DIRNAME)),
    'bet_file': (
      '{}/data/ilsvrc12/imagenet.bet.pickle'.format(REPO_DIRNAME)),
  }
  for key, val in default_args.iteritems():
    if not os.path.exists(val):
      raise Exception(
        "File for {} is missing. Should be at: {}".format(key, val))
  for key, val in googlenet_args.iteritems():
    if not os.path.exists(val):
      raise Exception(
        "File for {} is missing. Should be at: {}".format(key, val))

  default_args['image_dim'] = 256
  default_args['raw_scale'] = 255.

  # reference database
  database_param = '%s' % DATABASE_FILENAME

  def __init__(self, model_def_file, pretrained_model_file, mean_file,
         raw_scale, class_labels_file, bet_file, image_dim, gpu_mode):
    logging.info('Loading net and associated files...')
    if gpu_mode: caffe.set_mode_gpu()
    else: caffe.set_mode_cpu()

    ## load models
    # vgg16
    self.net = caffe.Classifier(
      model_def_file, pretrained_model_file,
      image_dims=(image_dim, image_dim), raw_scale=raw_scale,
      mean=np.array([103.939, 116.779, 123.68]), channel_swap=(2, 1, 0))
    logging.info('Load vision model, %s', model_def_file)
    # googlenet
    self.net_google = caffe.Classifier( self.googlenet_args['model_def_file'], 
      self.googlenet_args['pretrained_model_file'], 
      image_dims=(image_dim, image_dim), raw_scale=raw_scale, 
      mean=np.float32([104.0, 116.0, 122.0]), channel_swap=(2, 1, 0))
    logging.info('Load vision model, %s', self.googlenet_args['model_def_file'])

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

    self.bet = cPickle.load(open(bet_file))
    # A bias to prefer children nodes in single-chain paths
    # I am setting the value to 0.1 as a quick, simple model.
    # We could use better psychological models here...
    self.bet['infogain'] -= np.array(self.bet['preferences']) * 0.1


  def enroll_image(self, image, filename):
    try:
      # predict
      scores = self.net.predict([image], oversample=True).flatten()
      scores_google = self.net_google.predict([image], oversample=True).flatten()
      # extract features for retrieval
      logging.info('fc6 shape: {}'.format(self.net.blobs['fc6'].data.shape))
      logging.info('pool5/7x7_s1 shape: {}'.format(self.net_google.blobs['pool5/7x7_s1'].data.shape))
      feat_vgg = np.reshape(self.net.blobs['fc6'].data, (1,10*4096))
      feat_google = np.reshape(np.squeeze(self.net_google.blobs['pool5/7x7_s1'].data), (1,10*1024))
      feat = np.hstack((feat_vgg,feat_google))
      logging.info('feat shape: {}'.format(feat.shape))
      # binalize and 16bit-bitpacking
      fea = (np.hstack((np.packbits(np.uint8(feat > 0), axis=1)))).astype(np.uint16)
      fea_shift = fea << 8
      fea = fea_shift[0::2] + fea[1::2]
      enrolled_img_path = '/enroll/' + filename
      self.database['dic_ref'][self.database['ref'].shape[0]] = str(enrolled_img_path)
      self.database['dic_idx'][enrolled_img_path] = self.database['ref'].shape[0]
      self.database['ref'] = np.vstack((self.database['ref'], fea))
      logging.info("query shape: {}".format(fea.shape))
      logging.info('ref shape: {}'.format(self.database['ref'].shape))
      logging.info('Enroll done')
    except Exception as err:
      logging.info('Enroll error: %s', err)
      return (False, 'Something went wrong when enrolling the ' 'image. Maybe try another one?')
      

  def classify_image(self, image):
    try:
      # inference
      starttime = time.time()
      scores = self.net.predict([image], oversample=True).flatten()
      scores_google = self.net_google.predict([image], oversample=True).flatten()
      endtime = time.time()
      logging.info('Predict done for %d classes in %f', scores.shape[0], endtime - starttime)

      # score concate
      scores = np.vstack((scores, scores_google))
      scores = np.mean(scores, axis=0)
      # sort top-5 label
      indices = (-scores).argsort()[:5]
      predictions = self.labels[indices]
  
      # extract features for retrieval
      logging.info('fc6 shape: {}'.format(self.net.blobs['fc6'].data.shape))
      logging.info('pool5/7x7_s1 shape: {}'.format(self.net_google.blobs['pool5/7x7_s1'].data.shape))

      starttime = time.time()
      feat_vgg = np.reshape(self.net.blobs['fc6'].data, (1,10*4096))
      feat_google = np.reshape(np.squeeze(self.net_google.blobs['pool5/7x7_s1'].data), (1,10*1024))
      feat = np.hstack((feat_vgg,feat_google))
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
      for n, neighbor in enumerate(neighbor_list[0:NUM_NEIGHBOR]):
        logging.info("top-{}: {}, {}".format(n, self.database['dic_ref'][neighbor], dist[neighbor]))
        # ukb
        #result_neighbor.append('10.202.211.120:2596/PBrain/ukbench/full/%s' % (self.database['dic_ref'][neighbor]))
        # holidays
        #result_neighbor.append('10.202.211.120:2596/PBrain/holidays/jpg/%s' % (self.database['dic_ref'][neighbor]))
        # CDVS_Dataset
        result_neighbor.append('10.202.211.120:2596/PBrain/CDVS_Dataset/%s' % (self.database['dic_ref'][neighbor]))

      # In addition to the prediction text, we will also produce
      # the length for the progress bar visualization.
      meta = [
        (p, '%.5f' % scores[i])
        for i, p in zip(indices, predictions)
      ]
      logging.info('result: %s', str(meta))

      # Compute expected information gain
      expected_infogain = np.dot( self.bet['probmat'], scores[self.bet['idmapping']])
      expected_infogain *= self.bet['infogain']

      # sort the scores
      infogain_sort = expected_infogain.argsort()[::-1]
      bet_result = [(self.bet['words'][v], '%.5f' % expected_infogain[v]) for v in infogain_sort[:5]]
      logging.info('bet result: %s', str(bet_result))

      return (True, meta, bet_result, '%.3f' % (endtime - starttime), str(''), result_neighbor)

    except Exception as err:
      logging.info('Classification error: %s', err)
      return (False, 'Something went wrong when classifying the ' 'image. Maybe try another one?')


def start_tornado(app, port=5000):
  http_server = tornado.httpserver.HTTPServer(tornado.wsgi.WSGIContainer(app))
  http_server.listen(port)
  print("Tornado server starting on port {}".format(port))
  tornado.ioloop.IOLoop.instance().start()


def start_from_terminal(app):
  """
  Parse command line options and start the server.
  """
  parser = optparse.OptionParser()
  parser.add_option(
    '-d', '--debug',
    help="enable debug mode",
    action="store_true", default=False)
  parser.add_option(
    '-p', '--port',
    help="which port to serve content on",
    type='int', default=5000)
  parser.add_option(
    '-g', '--gpu',
    help="use gpu mode",
    action='store_true', default=False)

  opts, args = parser.parse_args()
  ImagenetClassifier.default_args.update({'gpu_mode': opts.gpu})

  # Initialize classifier + warm start by forward for allocation
  app.clf = ImagenetClassifier(**ImagenetClassifier.default_args)
  app.clf.net.forward()

  if opts.debug:
    app.run(debug=True, host='10.202.35.109', port=opts.port)
  else:
    start_tornado(app, opts.port)


if __name__ == '__main__':
  logging.getLogger().setLevel(logging.INFO)
  if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)
  start_from_terminal(app)
