
import numpy as np
import PIL.Image

def preprocess(net, img):
  img = np.float32(img)
  #if nd_image.ndim == 2:
  #  #import pdb; pdb.set_trace()
  #  nd_image = np.dstack((nd_image,nd_image))
  #  nd_image = np.dstack((nd_image,nd_image[:,:,0]))
  return np.rollaxis(img, 2)[::-1] - net.transformer.mean['data']


def deprocess(net, img):
  return np.dstack((img + net.transformer.mean['data'])[::-1])


def check_image_mode(img):
  if img.mode == 'RGB':
    return img
  elif img.mode == 'CMYK':
    img = img.convert('RGB')
    return img
  elif img.mode in ('RGBA', 'LA') or (img.mode == 'P' and 'transparency' in img.info):
    import pdb; pdb.set_trace()
    img.load()
    bg = PIL.Image.new("RGB", img.size, (255, 255, 255))
    bg.paste(img, mask=img.split()[-1])
    return bg
  elif img.mode == 'L' or img.mode == '1' or img.mode == 'I' or img.mode == 'F' or img.mode == 'P':
    npImg1C = np.asarray(img).astype(np.uint8)
    img = np.tile(npImg1C[:,:,np.newaxis] , (1,1,3)) #Broadcast 
    return PIL.Image.fromarray(img)
  else:
    #import pdb; pdb.set_trace()
    return None


def load_image(image_path):
  im = PIL.Image.open(image_path)
  im = check_image_mode( im )
  return im


def oversample(images, crop_dims):
  """
  Crop images into the four corners, center, and their mirrored versions.

  Parameters
  ----------
  image : (k x H x W) ndarrays
  crop_dims : (height, width) tuple for the crops.

  Returns
  -------
  crops : (10 x K x H x W) ndarray of crops
  """
  # Dimensions and center.
  im_shape = np.array(images.shape)
  crop_dims= np.array(crop_dims)
  im_center= im_shape[1:3] / 2.0

  # Make crop coordinates
  h_indices= (0, im_shape[1] - crop_dims[0])
  w_indices= (0, im_shape[2] - crop_dims[1])
  crops_ix = np.empty((5, 4), dtype=int)
  curr = 0
  for i in h_indices:
    for j in w_indices:
      crops_ix[curr] = (i, j, i + crop_dims[0], j + crop_dims[1])
      curr += 1
  crops_ix[4] = np.tile(im_center, (1, 2)) + np.concatenate([ -crop_dims / 2.0, crop_dims / 2.0 ])
  crops_ix = np.tile(crops_ix, (2, 1))

  # Extract crops
  crops = np.empty((10, im_shape[0], crop_dims[0], crop_dims[1]), dtype=np.float32)
  ix = 0
  #for im in images:
  #  for crop in crops_ix:
  #    crops[ix] = im[:, crop[0]:crop[2], crop[1]:crop[3]]
  #    ix += 1
  #  crops[ix-5:ix] = crops[ix-5:ix, :, :, ::-1]  # flip for mirrors
  for crop in crops_ix:
    crops[ix] = images[:, crop[0]:crop[2], crop[1]:crop[3]]
    ix += 1
  crops[ix-5:ix] = crops[ix-5:ix, :, :, ::-1]  # flip for mirrors
  return crops

