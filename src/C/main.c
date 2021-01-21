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
#define NH 64
#define NB 1

#define DATAPOINTS 60000

#define echo(x) do { puts(#x); (x); } while (0)

int main(int argc, char **argv) {

  u8 *data = malloc(sizeof(u8) * DATAPOINTS);
  assert(0==load("./data/train-images-idx3-ubyte", 16, DATAPOINTS, data));

  mat h, H, w, x, n, hn, err;
  mat negprods, posprods;

  mat_alloc(&h,   NH, NB); // hidden prob
  mat_alloc(&H,   NH, NB); // hidden state
  mat_alloc(&hn,  NH, NB); // hidden state neg
  mat_alloc(&w,   NX, NH); // vis-hid params
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

  mat_randn(&w, 0, 1);
  echo(mat_print(&w));

  float loss = 0.0f;
  float smloss = 0.0f;

  int ii = 0;
  //do {

    int r = 0;
    for (int k=0; k<NX; k++) x.data[k] = (float)(data[r*NX+k]) / 255.0f;
    echo(imshow(&x, 28));

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

    loss = mat_sum(&err) / (float)NB;
    smloss = ii == 0 ? loss : smloss * 0.999f + loss * 0.001f;

    printf("%.3f (%.3f)\n", loss, smloss);
    //poshidact = //
    //posvisact = //
    //neghidact = //
    //negcisact = //
    ii++;
  //} while (1);

  //dw = momentum *dw + eta(posprods - negprods)/NB;
  //w = w * (1-decay) + dw

  mat_free(&h), mat_free(&H); mat_free(&hn);
  mat_free(&w), mat_free(&x), mat_free(&n);
  mat_free(&posprods); mat_free(&negprods);
  mat_free(&err);

  free(data);
  return 0;
}
