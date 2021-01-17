import numpy as np

M = 4
N = 4
K = 4

if __name__ == "__main__":

  x = np.fromfile(open('x.bin'), dtype='float32')
  w = np.fromfile(open('w.bin'), dtype='float32')
  h = np.fromfile(open('h.bin'), dtype='float32')

  x = x.reshape(K,N)
  w = w.reshape(K,M)
  h = h.reshape(M,N)

  h0 = np.dot(w, x.T)

  print(h0.T)
  print(h)
