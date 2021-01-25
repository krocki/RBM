#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <assert.h>
#include <string.h>
#include "mat.h"
#include "util.h"
#include "io.h"

typedef uint8_t u8;

#define NX 784
#define NH 256
#define NB 32

#define DATAPOINTS 60000

#define echo(x) do { puts(#x); (x); } while (0)

int main(int argc, char **argv) {

  u8 *data = malloc(sizeof(u8) * NX * DATAPOINTS);
  assert(0==load("./data/train-images-idx3-ubyte", 16, NX * DATAPOINTS, data));

  mat h, H, w, x, n, hn, err, dw, dw0;
  mat negprods, posprods;

  mat_alloc(&h,   NH, NB); // hidden prob
  mat_alloc(&H,   NH, NB); // hidden state
  mat_alloc(&hn,  NH, NB); // hidden state neg
  mat_alloc(&w,   NX, NH); // vis-hid params
  mat_alloc(&dw,  NX, NH); // vis-hid param grads
  mat_alloc(&dw0, NX, NH); // vis-hid param grads
  mat_alloc(&x,   NX, NB); // data pos
  mat_alloc(&n,   NX, NB); // data neg (prob)
  mat_alloc(&err, NX, NB); // error

  mat_alloc(&posprods, NX, NH);
  mat_alloc(&negprods, NX, NH);

  mat_zero(&posprods); mat_zero(&negprods);
  mat_zero(&h), mat_zero(&w), mat_zero(&x);
  mat_zero(&H), mat_zero(&n), mat_zero(&hn);
  mat_zero(&err);
  mat_zero(&x);
  mat_zero(&dw);
  mat_zero(&dw0);

  mat_randn(&w, 0, 1e-1f);

  float loss = 0.0f;
  float smloss = 0.0f;
  float eta = 1e-3f;
  float momentum = 0.9f;

  int ii = 0;

  do {

    // prepare a batch
    for (int rr=0; rr<NB; rr++) {
      int r = rand() % DATAPOINTS;
      for (int k=0; k<NX; k++) x.data[k*NB+rr] = (float)(data[r*NX+k]) / 255.0f;
    }

    /* UP */
    mmul(&h, &w, &x, 1, 0);
    mat_apply(&h, &h, logistic);
    mat_apply(&H, &h, binarize);

    /* DOWN */
    mmul(&n, &w, &H, 0, 0);
    mat_apply(&n, &n, logistic);

    /* UP */
    mmul(&hn, &w, &n, 1, 0);
    mat_apply(&hn, &hn, logistic);

    mmul(&posprods, &x, &h, 0, 1);
    mmul(&negprods, &n, &hn, 0, 1);

    mat_sub(&err, &n, &x);
    mat_apply(&err, &err, square);

    // `grads`
    // dw = momentum *dw + eta(posprods - negprods)/NB;
    mat_sub(&dw0, &posprods, &negprods);
    mat_axpy(&dw, momentum, (1-momentum), &dw0);
    // w += w * eta + dw
    mat_axpy(&w, 1.0f, eta/NB, &dw);

    loss = mat_sum(&err) / (float)NB;
    smloss = ii == 0 ? loss : smloss * 0.999f + loss * 0.001f;

    if (ii%1000 == 0) {
      printf("ii=%d, loss=%7.3f, smloss=%7.3f, wnorm=%7.3f, dwnorm=%7.3f\n",
      ii, loss, smloss, mat_norm(&w), mat_norm(&dw));
      if (ii%10000 == 0) {
        mat_dump(&x, "x.bin");
        mat_dump(&n, "n.bin");
        mat_dump(&h, "h.bin");
        mat_dump(&H, "H.bin");
        mat_dump(&w, "w.bin");
        mat_dump(&dw, "dw.bin");
      }
    }

    ii++;

  } while (1);


  mat_free(&h), mat_free(&H); mat_free(&hn);
  mat_free(&w), mat_free(&x), mat_free(&n);
  mat_free(&posprods); mat_free(&negprods);
  mat_free(&err); mat_free(&dw); mat_free(&dw0);

  free(data);
  return 0;
}
