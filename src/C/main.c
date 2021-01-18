#include <stdio.h>
#include "mat.h"

#define NX 16
#define NH 4
#define NB 2

int main(int argc, char **argv) {

  mat h, w, x, posprods;

  mat_alloc(&h, NH, NB);
  mat_alloc(&w, NX, NH);
  mat_alloc(&x, NX, NB);
  mat_alloc(&posprods, NX, NH);

  mat_zero(&h), mat_zero(&w), mat_zero(&x);

  mat_randn(&w, 0, 1), mat_randf(&x);
  mat_print(&w), mat_print(&x);

  /* UP */
  mmul(&h, &w, &x, 1, 0);
  mat_apply(&h, &h, logistic);
  mmul(&posprods, &x, &h, 0, 1);

  mat_dump(&h, "h.bin");
  mat_dump(&w, "w.bin");
  mat_dump(&x, "x.bin");
  mat_dump(&posprods, "posprods.bin");

  mat_print(&h);
  mat_print(&posprods);

  mat_free(&h), mat_free(&w), mat_free(&x);
  mat_free(&posprods);

  return 0;
}
