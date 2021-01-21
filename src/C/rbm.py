import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')

NX = 784
NH = 64
D = int(np.sqrt(NH))

NB = 8
sigma = 1e-2
eta = 1e-3
decay = 1e-4
momentum = .9
smoothing = 1e-3
smerr = None
smact = None

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

  w = np.random.randn(NX, NH).astype(np.float32) * sigma
  b = np.zeros((NH, 1), dtype=np.float32)
  c = np.zeros((NX, 1), dtype=np.float32)

  dw = np.zeros_like(w)
  db = np.zeros_like(b)
  dc = np.zeros_like(c)

  data, _ = read_mnist();

  ii = 0

  while True:

    idx = np.random.randint(60000-NB+1)
    x = data[idx:idx+NB, :].T

    # up
    h = np.dot(w.T, x) + b
    h = logistic(h)

    # down
    r = np.random.rand(NH, NB).astype(np.float32)
    H = (r < h).astype(np.float32)
    n = np.dot(w, H) + c
    n = logistic(n)

    # up
    hn = np.dot(w.T, n) + b
    hn = logistic(hn)

    # change w
    posprods = np.dot(x, h.T)
    negprods = np.dot(n, hn.T)

    poshidact = np.sum(h)/NB
    posvisact = np.sum(x)/NB
    neghidact = np.sum(hn)/NB
    negvisact = np.sum(n)/NB

    err = np.sum(((x - n)**2)/NB)

    smerr = err if smerr is None else smerr * (1-smoothing) + err * smoothing
    smact = poshidact if smact is None else smact * (1-smoothing) + poshidact * smoothing

    if 0==ii%10000 and ii>0:

      print(f"{ii}: err={smerr:.3f} act={smact/NH:.3f}")

      im_x = np.transpose(x.reshape(28, 28, NB), (0, 2, 1)).reshape(28, 28*NB)
      im_n = np.transpose(n.reshape(28, 28, NB), (0, 2, 1)).reshape(28, 28*NB)
      w_ = w.copy()
      w_nrm = np.linalg.norm(w_, axis=0, keepdims=True)
      w_ /= w_nrm
      w_nrm = np.linalg.norm(w_, axis=0, keepdims=True)
      im_w = np.transpose(w_.reshape(28, 28, D, D), (2, 0, 3, 1)).reshape(28*D, 28*D)
      im = np.vstack((im_x, im_n))

      np_save_img(im, f"im.png")
      np_save_img(im_w, f"w.png")
      np_save_img(np.transpose(h.reshape(D, D, NB), (2, 0, 1)).reshape(D*NB, D), f"h_prob.png")
      np_save_img(np.transpose(H.reshape(D, D, NB), (2, 0, 1)).reshape(D*NB, D), f"h_state.png")

    # adjust w
    dw = momentum * dw + eta*(posprods - negprods)/NB
    db = momentum * db + eta*(poshidact - neghidact)/NB
    dc = momentum * dc + eta*(posvisact - negvisact)/NB

    w = w * (1 - decay) + dw
    b = b * (1 - decay) + db
    c = c * (1 - decay) + dc

    ii += 1
