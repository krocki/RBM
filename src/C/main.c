#include <stdio.h>
#include "mat.h"

#define M 4
#define N 4
#define K 4

int main(int argc, char **argv) {

  mat h, w, x;
  mat_alloc(&h, M, N);
  mat_alloc(&w, K, M);
  mat_alloc(&x, K, N);

  mat_zero(&h), mat_zero(&w), mat_zero(&x);

  mat_randn(&w, 0, 1), mat_randf(&x);
  mat_print(&w), mat_print(&x);

  /* h := w'x */
  mmul(&h, &w, &x, 1, 0);

  mat_dump(&h, "h.bin");
  mat_dump(&w, "w.bin");
  mat_dump(&x, "x.bin");

  mat_print(&h);
  mat_free(&h), mat_free(&w), mat_free(&x);

  return 0;
}