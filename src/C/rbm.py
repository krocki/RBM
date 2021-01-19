import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')

NX = 784
NH = 100
D = int(np.sqrt(NH))

NB = 8
sigma = 1e-2
eta = 1e-4
decay = 1e-5
smerr = None

def np_debug_print(x):

  np.set_printoptions(precision=3)
  print(f'shape: {x.shape}')
  print(f'\t' + str(x).replace('\n', '\n\t'))
  print(f"===============")

def np_save_img(arr, fname, cmap='viridis'):

  p = plt.imshow(arr, cmap=cmap, interpolation='nearest')
  plt.xticks([]), plt.yticks([])
  plt.savefig(f'{fname}')
  plt.close('all')

def read_mnist():

  f = open('./data/train-images-idx3-ubyte')
  raw = np.fromfile(file=f, dtype=np.uint8)
  data = raw[16:].reshape((60000, 784)).astype(np.float32) / 255
  f.close()

  f = open('./data/train-labels-idx1-ubyte')
  raw = np.fromfile(file=f, dtype=np.uint8)
  labels = raw[8:].reshape((60000)).astype(np.int32)
  f.close()

  return data, labels

def logistic(x):
  return 1. / (1. + np.exp(-x))

if __name__ == "__main__":

  np.set_printoptions(precision=3)
  plt.ion()

  w = np.random.randn(NX, NH).astype('float32') * sigma
  data, _ = read_mnist();

  ii = 0

  while True:

    idx = np.random.randint(60000-NB+1)
    x = data[idx:idx+NB, :].T

    # positive
    h = np.dot(w.T, x)
    h = logistic(h)

    # down
    r = np.random.rand(NH, NB)
    H = (r < h).astype(np.float32)
    n = np.dot(w, h)

    # up
    hn = np.dot(w.T, n)
    hn = logistic(hn)

    # change w
    posprods = np.dot(x, h.T)
    negprods = np.dot(n, hn.T)

    err = np.linalg.norm(x - n)/NB
    smerr = err if smerr is None else smerr * 0.99 + err * 0.01

    if 0==ii%10000 and ii>0:

      print(f"{ii}: {smerr}")

      im_x = np.transpose(x.reshape(28, 28, NB), (0, 2, 1)).reshape(28, 28*NB)
      im_n = np.transpose(n.reshape(28, 28, NB), (0, 2, 1)).reshape(28, 28*NB)
      im_w = np.transpose(w.reshape(28, 28, D, D), (2, 0, 3, 1)).reshape(28*D, 28*D)
      im = np.vstack((im_x, im_n))

      np_save_img(im, f"im.png")
      np_save_img(im_w, f"w.png")
      np_save_img(np.transpose(h.reshape(D, D, NB), (2, 0, 1)).reshape(D*NB, D), f"h.png")
      np_save_img(np.transpose(H.reshape(D, D, NB), (2, 0, 1)).reshape(D*NB, D), f"H.png")

    # adjust w
    w_delta = (posprods - negprods)
    w *= (1-decay)
    w += w_delta * eta

    ii += 1
