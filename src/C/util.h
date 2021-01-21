void imshow(mat *m, int cols_per_row) {

  int n = m->r * m->c;
  float v;
  for (int i=0; i<n; i++) {
    v = m->data[i];
    printf("%c%s",
      v>=0.75f ? '@' :
      v>=0.50f ? 'O' :
      v>=0.25f ? 'o' :
      '.',
      (i+1)%(cols_per_row) ? "" : "\n");
  }

}
