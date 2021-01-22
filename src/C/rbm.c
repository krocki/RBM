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
#define NH 100
#define NB 1

#define DATAPOINTS 60000

#define echo(x) do { puts(#x); (x); } while (0)

int main(int argc, char **argv) {

  u8 *data = malloc(sizeof(u8) * NX * DATAPOINTS);
  assert(0==load("./data/train-images-idx3-ubyte", 16, NX * DATAPOINTS, data));

  mat h, H, w, x, n, hn, err, dw;
  mat negprods, posprods;

  mat_alloc(&h,   NH, NB); // hidden prob
  mat_alloc(&H,   NH, NB); // hidden state
  mat_alloc(&hn,  NH, NB); // hidden state neg
  mat_alloc(&w,   NX, NH); // vis-hid params
  mat_alloc(&dw,  NX, NH); // vis-hid param grads
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

  mat_randn(&w, 0, 1e-1);
  //echo(mat_print(&w));

  float loss = 0.0f;
  float smloss = 0.0f;
  float eta = 1e-2f;

  int ii = 0;

  do {

    int r = rand() % 100;
    for (int k=0; k<NX; k++) x.data[k] = (float)(data[r*NX+k]) / 255.0f;

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
    mmul(&negprods, &n, &h, 0, 1);

    mat_sub(&err, &x, &n);
    mat_apply(&err, &err, square);

    // `grads`
    // dw = momentum *dw + eta(posprods - negprods)/NB;
    mat_sub(&dw, &posprods, &negprods);
    // w += w * eta + dw
    mat_axpy(&w, eta/NB, &dw);

    loss = mat_sum(&err) / (float)NB;
    smloss = ii == 0 ? loss : smloss * 0.999f + loss * 0.001f;

    if (ii%1000 == 0) {
      printf("loss=%.3f, smloss=%.3f, wnorm=%.3f, dwnorm=%.3f\n",
      loss, smloss, mat_norm(&w), mat_norm(&dw));
      imshow(&n, 28);
    }

    ii++;

  } while (1);


  mat_free(&h), mat_free(&H); mat_free(&hn);
  mat_free(&w), mat_free(&x), mat_free(&n);
  mat_free(&posprods); mat_free(&negprods);
  mat_free(&err); mat_free(&dw);

  free(data);
  return 0;
}
