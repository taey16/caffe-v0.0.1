from cStringIO import StringIO
import numpy as np
import scipy.ndimage as nd
import PIL.Image
from IPython.display import clear_output, Image, display
from google.protobuf import text_format

import caffe

def preprocess(net, img):
    return np.float32(np.rollaxis(img, 2)[::-1]) - net.transformer.mean['data']
def deprocess(net, img):
    return np.dstack((img + net.transformer.mean['data'])[::-1])
def showarray(a, fmt='jpeg'):
    a = np.uint8(np.clip(a, 0, 255))
    f = StringIO()
    PIL.Image.fromarray(a).save(f, fmt)
    display(Image(data=f.getvalue()))

    return Image(data=f.getvalue())

def objective_L2(dst):
    dst.diff[:] = dst.data 

def make_step(net, step_size=1.5, end='inception_4c/output', jitter=32, clip=True, objective=objective_L2):
    '''Basic gradient ascent step.'''

    src = net.blobs['data'] # input image is stored in Net's 'data' blob
    dst = net.blobs[end]

    ox, oy = np.random.randint(-jitter, jitter+1, 2)
    src.data[0] = np.roll(np.roll(src.data[0], ox, -1), oy, -2) # apply jitter shift
            
    net.forward(end=end)
    # specify the optimization objective
    objective(dst)
    net.backward(start=end)
    g = src.diff[0]
    # apply normalized ascent step to the input image
    src.data[:] += step_size/np.abs(g).mean() * g

    src.data[0] = np.roll(np.roll(src.data[0], -ox, -1), -oy, -2) # unshift image
            
    if clip:
        bias = net.transformer.mean['data']
        src.data[:] = np.clip(src.data, -bias, 255-bias)    

def deepdream(net, base_img, iter_n=10, octave_n=4, octave_scale=1.4, end='inception_4c/output', clip=True, **step_params):
    # prepare base images for all octaves
    octaves = [preprocess(net, base_img)]
    for i in xrange(octave_n-1):
        octaves.append(nd.zoom(octaves[-1], (1, 1.0/octave_scale,1.0/octave_scale), order=1))
    
    src = net.blobs['data']
    L = 0
    # allocate image for network-produced details
    detail = np.zeros_like(octaves[-1])
    for octave, octave_base in enumerate(octaves[::-1]):
        h, w = octave_base.shape[-2:]
        if octave > 0:
            # upscale details from the previous octave
            h1, w1 = detail.shape[-2:]
            detail = nd.zoom(detail, (1, 1.0*h/h1,1.0*w/w1), order=1)

        # resize the network's input image size
        src.reshape(1,3,h,w)
        src.data[0] = octave_base+detail
        for i in xrange(iter_n):
            make_step(net, end=end, clip=clip, **step_params)
            
            # visualization
            vis = deprocess(net, src.data[0])
            # adjust image contrast if clipping is disabled
            if not clip:
                vis = vis*(255.0/np.percentile(vis, 99.98))
            showarray(vis)

            # compute loss
            num_activation_neuron = net.blobs[end].data.size
            activation = net.blobs[end].data.reshape( num_activation_neuron, 1)
            loss = 0.5 * (np.power(np.linalg.norm(activation, 2), 2) ) / num_activation_neuron
            L = np.append(L, loss)
            print '(octave: %d, iter: %d)' %(octave, i), ', output: %s' %(end), ', octave shape: (%d,%d,%d)' %(vis.shape[0], vis.shape[1], vis.shape[2]), ', loss: %f' %(L[-1])
            clear_output(wait=True)
            
        # extract details produced on the current octave
        detail = src.data[0]-octave_base
    # returning the resulting image
    return deprocess(net, src.data[0])

def deconv(net, base_img, end='inception_4c/output'):
    input_img = preprocess(net, base_img)
    src = net.blobs['data']
    dst = net.blobs[end]
    h, w = src.data.shape[-2:]
    src.reshape(1, 3, h, w)
    src.data[0] = input_img
    net.forward(end=end)
    dst.diff[:] = dst.data[0]
    net.backward(start=end)
    vis = deprocess(net, src.diff[0])
    #bias = net.transformer.mean['data']
    #reconstructed = np.clip(vis, -bias, 255-bias)

    return vis

if __name__ == "__main__":

    model_path = '../../caffe/models/bvlc_googlenet/'
    net_fn   = model_path + 'deploy.prototxt'
    param_fn = model_path + 'bvlc_googlenet.caffemodel'

    # Patching model to be able to compute gradients.
    # Note that you can also manually add "force_backward: true" line to "deploy.prototxt".
    model = caffe.io.caffe_pb2.NetParameter()
    text_format.Merge(open(net_fn).read(), model)
    model.force_backward = True
    open('tmp.prototxt', 'w').write(str(model))

    net = caffe.Classifier('tmp.prototxt', param_fn, mean = np.float32([104.0, 116.0, 122.0]), channel_swap = (2,1,0)) 

    import pdb; pdb.set_trace()
    resized = 224, 224
    img = PIL.Image.open('/Users/1002596/Documents/caffe/examples/images/cat.jpg');
    img = img.resize( resized, PIL.Image.ANTIALIAS )
    img = np.float32(img)
    recog = showarray(deconv(net, img))
    #_,L=deepdream(net, img, iter_n=69, end='inception_4c/output')
    dmy=1

