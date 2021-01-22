typedef struct {
  float *data;
  int r, c, n;
  enum { col_major, row_major } order;
} mat;

void  mat_alloc(mat *m, int r, int c);
void  mat_free(mat *m);
void  mat_zero(mat *m);
void  mat_print(mat *m);
void  mat_dump(mat *b, const char *);
void  mat_apply(mat *out, mat *in, float (*func)(float));
void  mat_sub(mat *out, mat *in0, mat *in1);
float mat_norm(mat *a);
float mat_sum(mat *a);
float mat_min(mat *a);
float mat_max(mat *a);
// y := a*x + y
void  mat_axpy(mat *y, float a, mat *x);

void  mmul(mat *c, const mat *restrict a, const mat *restrict b, int, int);

float randn(float mean, float stddev);
float logistic(float);
float randf();
float binarize(float);
float square(float);

void mat_randn(mat *m, float mean, float std);
void mat_randf(mat *m);
void mat_logistic(mat *m);
