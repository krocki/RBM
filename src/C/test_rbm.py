import numpy as np

NX = 16
NH = 4
NB = 2

if __name__ == "__main__":

  np.set_printoptions(precision=3)

  x = np.fromfile(open('x.bin'), dtype='float32')
  w = np.fromfile(open('w.bin'), dtype='float32')
  h = np.fromfile(open('h.bin'), dtype='float32')

  x = x.reshape(NX, NB)
  w = w.reshape(NX, NH)
  h = h.reshape(NH, NB)

  print('x'); print(x.shape); print(x)
  print('w'); print(w.shape); print(w)
  print('h'); print(h.shape); print(h)

  h0 = np.dot(w.T, x)

  print('h_py'); print(h0.shape); print(h0)
