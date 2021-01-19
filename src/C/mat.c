#include "mat.h"
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <assert.h>
#include <stdarg.h>

#define USE_BLAS 1
#if USE_BLAS
#ifdef USE_MKL
#include <mkl_cblas.h>
#else
#include <cblas.h>
#endif
#endif

void mat_alloc(mat *m, int r, int c) {
  m->data = calloc(r * c, sizeof(float));
  m->r = r; m->c = c;
  m->n = r * c;
  m->order = row_major;
}

void mat_free(mat *m) {

  assert(NULL != m); assert(NULL != m->data);
  free(m->data); m->data = NULL;
  m->r = 0; m->c = 0;

}

int print_f(float *f) {
  if (fabs(*f)>1e-6) {
    printf("%6.3f", *f);
    return 1;
  }
  else {
    printf("  --- ");
    return 0;
  }
}

/*
   void mat_print(mat *m) {
  printf("===== %d %d %s =====\n",
  m->r, m->c,
  m->order == col_major ?
  "col_major" : "row_major");

  printf(" ");
  for (size_t i=0; i<m->c; i++)
    printf("%6zu ", i);
  printf("\n");

  int nnz = 0;
  for (size_t j=0; j<m->r; j++) {
    printf("%2zu ", j);
    for (int i=0; i<m->c; i++) {
      nnz+=print_f(&(m->data[i*m->r+j]));
      if (i==(m->c-1)) {
        printf("%s", nnz==0 ? "\r" : "\n");
        nnz = 0;
      }
      else printf(" ");
    }
  }
}
*/
void mat_print(mat *m) {

  int r = m->order == col_major ? m->r : m->c;
  int c = m->order == col_major ? m->c : m->r;

  printf("===== %d %d %s =====\n",
  m->r, m->c, m->order == col_major ?
  "col_major" : "row_major");

  printf(" ");
  for (size_t i=0; i<r; i++)
    printf("%6zu ", i);
  printf("\n");

  int nnz = 0;
  for (size_t j=0; j<c; j++) {
    printf("%2zu ", j);
    for (int i=0; i<r; i++) {
      nnz+=print_f(&(m->data[m->order == col_major ? i*r+j : j*r+i]));
      if (i==(r-1)) {
        printf("%s", nnz==0 ? "\r" : "\n");
        nnz = 0;
      }
      else printf(" ");
    }
  }
}

int checknan(int n, float *arr) {
  for (int i=0; i<n; i++)
    if (!isnan(arr[i])) {
      puts("nan found\n");
      return 1;
    }
  return 0;
}

int nnz(int n, const float *a) {
  int o=0;
  for (int i=0; i<n; i++)
    if (fabs(a[i]) > 1e-9) o++;
  return o;
}

void sub(int n,
         float *c,
         const float *a,
         const float *b) {

  for (int i=0; i<n; i++)
    c[i] = a[i] - b[i];

}

float min(int n, const float *a) {
  float r = .0f;
  for (int i=0; i<n; i++)
    r = ((i==0) || r>a[i]) ? a[i] : r;
  return r;
}

float max(int n, const float *a) {
  float r = .0f;
  for (int i=0; i<n; i++)
    r = ((i==0) || r<a[i]) ? a[i] : r;
  return r;
}

float norm(int n, const float *a) {
  float r = .0f;
  for (int i=0; i<n; i++)
    r += a[i]*a[i];

  return sqrtf(r);
}

float logistic(float x) {
  return 1. / (1. + expf(-x));
}

float binarize(float x) {
  return randf() < x ? 1.f : 0.f;
}

void mat_zero(mat *m) {
  int n = m->r * m->c;
  memset(m->data, 0, n * sizeof(float));
}

void mat_randf(mat *m) {
  for (int i=0; i<(m->r * m->c); i++) {
    m->data[i] = randf();
  }
}

void mat_randn(mat *m, float mean, float stddev) {
  for (int i=0; i<(m->r * m->c); i++) {
    m->data[i] = randn(mean, stddev);
  }
}

float mat_norm(mat *m) { return norm(m->r * m->c, m->data); }
float mat_min(mat *m)  { return min(m->r * m->c, m->data); }
float mat_max(mat *m)  { return max(m->r * m->c, m->data); }

void mat_apply(mat *out, mat *in, float (*func)(float)) {

  int n_in = out->r * in->c;
  int n_out = in->r * in->c;
  assert(n_in == n_out);

  for (int i=0; i<n_out; i++) {
    out->data[i] = func(in->data[i]);
  }

}

void dump_arr_f(int n, float *m, const char *fn) {
  FILE *f = fopen(fn, "wb");
  if (!f) return;
  fwrite(m, n, sizeof(float), f);
  fclose(f);
}

void mat_dump(mat *m, const char *fn) {
  if (!m) return;
  size_t n = m->c * m->r;
  dump_arr_f(n, m->data, fn);
}

// non-blas
void mmul1(
        mat *restrict c,
  const mat *restrict a,
  const mat *restrict b,
  int transA, int transB);

// top-level
void mmul(
        mat *restrict c,
  const mat *restrict a,
  const mat *restrict b,
  int transA, int transB) {

#if USE_BLAS
//#ifdef USE_MKL
//  mkl_set_num_threads(1);
//#else
//  openblas_set_num_threads(1);
//#endif

  enum CBLAS_TRANSPOSE at =
    (transA) ? CblasTrans : CblasNoTrans;
  enum CBLAS_TRANSPOSE bt =
    (transB) ? CblasTrans : CblasNoTrans;

  int M = c->r, N = c->c;
  int K = transA ? a->r : a->c;

  float alpha = 1.f;
  float beta = 1.f;

  int lda = transA ? M : K;
  int ldb = transB ? K : N;
  int ldc = N;

  cblas_sgemm( c->order == col_major ? CblasColMajor : CblasRowMajor, at, bt, M, N, K,
               alpha,
               a->data, lda,
               b->data, ldb,
               beta,
               c->data, ldc );

#else
  // default impl
  mmul1(c, a, b, transA, transB);
#endif

}

void mmul1(mat * mc,
     const mat * ma,
     const mat * mb,
     int transA, int transB) {

  int M = mc->r, N = mc->c;
  int K = transA ? ma->r : ma->c ;

  float * restrict c = mc->data;
  float * restrict a = ma->data;
  float * restrict b = mb->data;

  //NN
  if (0==transA && 0==transB) {
    for (int i=0; i<M; i++)
    for (int j=0; j<N; j++)
    for (int k=0; k<K; k++)
      c[j*M+i] += a[k*M+i] * b[k+K*j];
  }

  //TN
  if (1==transA && 0==transB) {
    for (int j=0; j<N; j++)
    for (int i=0; i<M; i++)
    for (int k=0; k<K; k++)
      c[j*M+i] += a[k+K*i] * b[k+K*j];
  }

  //NT
  if (0==transA && 1==transB) {
    for (int i=0; i<M; i++)
    for (int j=0; j<N; j++)
    for (int k=0; k<K; k++)
      c[j*M+i] += a[k*M+i] * b[k*N+j];
  }

  //TT
  if (1==transA && 1==transB) {
    for (int j=0; j<N; j++)
    for (int i=0; i<M; i++)
    for (int k=0; k<K; k++)
      c[j*M+i] += a[k+K*i] * b[k*N+j];
  }
}
