#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <assert.h>
#include <string.h>
#include "mat.h"
#include "util.h"
#include "io.h"
#include "timer.h"

#if defined(__APPLE__) && defined(__MACH__)
#include <mach/mach_time.h>
#endif

typedef uint8_t u8;

#define NX 784
#define NH 100
#define NB 16

#define DATAPOINTS 60000

#define echo(x) do { puts(#x); (x); } while (0)

#define NO_TIMERS 8
#define TIMER_INTERVAL 10000
//double get_nsec(uint64_t t) {
//  mach_timebase_info_data_t info;
//  mach_timebase_info(&info);
//  return (double)(t * info.numer) / (double)info.denom;
//}

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
  float eta = 1e-4f;
  float momentum = 0.99f;
  double time_per_iter[NO_TIMERS] = {0};
  //uint64_t t[NO_TIMERS+1] = {0};
  struct timeval ts, te, t[NO_TIMERS+1];

  int ii = 0;
  gettimeofday(&ts, NULL);

  do {

//// T0 START //////////////////
    gettimeofday(&t[0], NULL);
    // prepare a batch
    for (int rr=0; rr<NB; rr++) {
      int r = rand() % DATAPOINTS;
      for (int k=0; k<NX; k++) x.data[k*NB+rr] = (float)(data[r*NX+k]) / 255.0f;
    }

//// T1 START //////////////////
    gettimeofday(&t[1], NULL);
    time_per_iter[0] += get_time_diff(&t[0], &t[1]);

    /* UP */
    mmul(&h, &w, &x, 1, 0);

//// T2 START //////////////////
    gettimeofday(&t[2], NULL);
    time_per_iter[1] += get_time_diff(&t[1], &t[2]);

    mat_apply(&h, &h, logistic);

//// T3 START //////////////////
    gettimeofday(&t[3], NULL);
    time_per_iter[2] += get_time_diff(&t[2], &t[3]);

    mat_apply(&H, &h, binarize);

//// T4 START //////////////////
    gettimeofday(&t[4], NULL);
    time_per_iter[3] += get_time_diff(&t[3], &t[4]);

    /* DOWN */
    mmul(&n, &w, &H, 0, 0);
    mat_apply(&n, &n, logistic);

    /* UP */
    mmul(&hn, &w, &n, 1, 0);
    mat_apply(&hn, &hn, logistic);

//// T5 START //////////////////
    gettimeofday(&t[5], NULL);
    time_per_iter[4] += get_time_diff(&t[4], &t[5]);

    mmul(&posprods, &x, &h, 0, 1);
    mmul(&negprods, &n, &hn, 0, 1);

    mat_sub(&err, &n, &x);
    mat_apply(&err, &err, square);

//// T6 START //////////////////
    gettimeofday(&t[6], NULL);
    time_per_iter[5] += get_time_diff(&t[5], &t[6]);

    // `grads`
    // dw = momentum *dw + eta(posprods - negprods)/NB;
    mat_sub(&dw0, &posprods, &negprods);
    mat_axpy(&dw, momentum, (1-momentum), &dw0);

//// T7 START //////////////////
    gettimeofday(&t[7], NULL);
    time_per_iter[6] += get_time_diff(&t[6], &t[7]);
    mat_axpy(&w, 1.0f, eta/NB, &dw);

//// T8 START //////////////////
    gettimeofday(&t[8], NULL);
    time_per_iter[7] += get_time_diff(&t[7], &t[8]);

    loss = mat_sum(&err) / (float)NB;
    smloss = ii == 0 ? loss : smloss * 0.999f + loss * 0.001f;

    if (0<ii && ii%1000 == 0) {
      gettimeofday(&te, NULL);
      printf("%9.3f s, ii=%d, loss=%7.3f, smloss=%7.3f, wnorm=%7.3f, dwnorm=%7.3f\n", get_time_diff(&ts, &te), ii, loss, smloss, mat_norm(&w), mat_norm(&dw));
      if (ii%TIMER_INTERVAL == 0) {

        mat_dump(&x, "x.bin");
        mat_dump(&n, "n.bin");
        mat_dump(&h, "h.bin");
        mat_dump(&H, "h_.bin");
        mat_dump(&w, "w.bin");
        mat_dump(&dw, "dw.bin");

        //for (int tt=0; tt<NO_TIMERS; tt++) {
          //printf("t[%d] = %12.6f s\n", tt, time_per_iter[tt] / (double)TIMER_INTERVAL);
          //time_per_iter[tt] = 0.0f;

        //}
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
