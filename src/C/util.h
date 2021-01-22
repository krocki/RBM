void imshow(mat *m, int cols_per_row) {

  int n = m->r * m->c;
  float v;
  for (int i=0; i<n; i++) {
    v = m->data[i];
    printf("%c%s",
      v>=0.9 ? '@' :
      v>=0.8 ? 'O' :
      v>=0.7 ? 'o' :
      '.',
      (i+1)%(cols_per_row) ? "" : "\n");
  }

}
