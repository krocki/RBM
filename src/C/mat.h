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
float mat_norm(mat *a);
float mat_min(mat *a);
float mat_max(mat *a);

void  mmul(mat *c, const mat *restrict a, const mat *restrict b, int, int);

float randn(float mean, float stddev);
float randf();

void mat_randn(mat *m, float mean, float std);
void mat_randf(mat *m);