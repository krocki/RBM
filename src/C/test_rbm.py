import numpy as np

NX = 16
NH = 4
NB = 2

if __name__ == "__main__":

  np.set_printoptions(precision=3)

  # C-saved files
  xC = np.fromfile(open('x.bin'), dtype='float32')
  wC = np.fromfile(open('w.bin'), dtype='float32')
  hC = np.fromfile(open('h.bin'), dtype='float32')

  xC = xC.reshape(NX, NB)
  wC = wC.reshape(NX, NH)
  hC = hC.reshape(NH, NB)

  w = wC.copy()
  x = xC.copy()

  print('xC'); print(xC.shape); print(xC)
  print('wC'); print(wC.shape); print(wC)
  print('hC'); print(hC.shape); print(hC)

  h = np.dot(w.T, x)

  print('h_py'); print(h.shape); print(h)
  print(f'norm(h-hC)={np.linalg.norm(h - hC)}')
  print(f'maxdiff(h-hC)={np.max(np.fabs(h - hC))}')
