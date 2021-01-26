import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')

NX = 784
NH = 100
D = int(np.sqrt(NH))
NB = 16
d = int(np.sqrt(NB))

def np_save_img(arr, fname, cmap='viridis'):

  p = plt.imshow(arr, cmap=cmap, interpolation='nearest')
  plt.xticks([]), plt.yticks([])
  plt.savefig(f'{fname}', dpi=300)
  plt.close('all')

def logistic(x):
  return 1. / (1. + np.exp(-x))

def vis_weights(w):
  w_ = w.copy()
  w_nrm = np.linalg.norm(w_, axis=0, keepdims=True)
  w_ /= w_nrm
  w_nrm = np.linalg.norm(w_, axis=0, keepdims=True)
  im_w = np.transpose(w_.reshape(28, 28, D, D), (2, 0, 3, 1)).reshape(28*D, 28*D)
  return im_w

if __name__ == "__main__":

  np.set_printoptions(precision=4)

  # C-saved files
  x = np.fromfile(open('x.bin'), dtype='float32')
  w = np.fromfile(open('w.bin'), dtype='float32')
  dw = np.fromfile(open('dw.bin'), dtype='float32')
  h = np.fromfile(open('h.bin'), dtype='float32')
  H = np.fromfile(open('h_.bin'), dtype='float32')
  n = np.fromfile(open('n.bin'), dtype='float32')

  x = x.reshape(NX, NB)
  n = n.reshape(NX, NB)
  w = w.reshape(NX, NH)
  dw = dw.reshape(NX, NH)
  h = h.reshape(NH, NB)
  H = H.reshape(NH, NB)

  im_x = np.transpose(x.reshape(28, 28, d, d), (3, 0, 2, 1)).reshape(28*d, 28*d)
  im_n = np.transpose(n.reshape(28, 28, d, d), (3, 0, 2, 1)).reshape(28*d, 28*d)
  im = np.hstack((im_x, im_n))
  im_w = vis_weights(w)
  im_dw = vis_weights(dw)

  np_save_img(im, f"im.png")
  np_save_img(im_w, f"w.png")
  np_save_img(im_dw, f"dw.png")
  np_save_img(np.transpose(h.reshape(D, D, NB), (2, 0, 1)).reshape(D*NB, D), f"h_prob.png")
  np_save_img(np.transpose(H.reshape(D, D, NB), (2, 0, 1)).reshape(D*NB, D), f"h_state.png")
