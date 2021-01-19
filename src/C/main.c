#include <stdio.h>
#include "mat.h"

#define NX 16
#define NH 4
#define NB 2

#define echo(x) do { puts(#x); (x); } while (0)

int main(int argc, char **argv) {

  mat h, H, w, x, v, hn;
  mat negprods, posprods;

  mat_alloc(&h, NH, NB); // hidden prob
  mat_alloc(&H, NH, NB); // hidden state
  mat_alloc(&hn, NH, NB); // hidden state neg
  mat_alloc(&w, NX, NH); // vis-hid params
  mat_alloc(&x, NX, NB); // data pos
  mat_alloc(&v, NX, NB); // data neg (prob)

  mat_alloc(&posprods, NX, NH);
  mat_alloc(&negprods, NX, NH);

  mat_zero(&posprods); mat_zero(&negprods);
  mat_zero(&h), mat_zero(&w), mat_zero(&x);
  mat_zero(&H), mat_zero(&v), mat_zero(&hn);

  mat_randn(&w, 0, 1), mat_randf(&x);
  mat_print(&w), mat_print(&x);

  /* UP */
  mmul(&h, &w, &x, 1, 0);
  mat_apply(&h, &h, logistic);
  mat_apply(&H, &h, binarize);

  mmul(&posprods, &x, &h, 0, 1);

  /* DOWN */
  mmul(&v, &w, &H, 0, 0);
  mat_apply(&v, &v, logistic);

  /* UP */
  mmul(&hn, &w, &v, 1, 0);
  mat_apply(&hn, &hn, logistic);
  mmul(&negprods, &v, &h, 0, 1);

  mat_dump(&h, "h.bin");
  mat_dump(&H, "H.bin");
  mat_dump(&w, "w.bin");
  mat_dump(&x, "x.bin");

  mat_dump(&posprods, "posprods.bin");
  mat_dump(&negprods, "negprods.bin");

  echo(mat_print(&h));
  echo(mat_print(&H));
  echo(mat_print(&v));
  echo(mat_print(&hn));
  echo(mat_print(&posprods));
  echo(mat_print(&negprods));

  mat_free(&h), mat_free(&H); mat_free(&hn);
  mat_free(&w), mat_free(&x), mat_free(&v);
  mat_free(&posprods); mat_free(&negprods);

  return 0;
}
