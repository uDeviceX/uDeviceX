static const int MAX_N = 999999;

void matrices_read(const char *path, /**/ Matrices **pq) {
    int n;
    FILE *f;
    Matrices *q;
    EMALLOC(1, &q);
    EMALLOC(MAX_N, &q->m);
    UC(efopen(path, "r", /**/ &f));
    UC(efclose(f));
    n = 0;
    //    while (read_matrix(f, /**/ q->m.) ) {
    q->n = n;

    *pq = q;
}

void matrices_fin(Matrices *q) { EFREE(q->m); EFREE(q); }
