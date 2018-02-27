static const int MAX_N = 999999;

static void matrix2r(const double A[16], double r[3]) {
    enum {X, Y, Z};
    int i;
    i = 0;
    i++; i++; i++; r[X] = A[i++];
    i++; i++; i++; r[Y] = A[i++];
    i++; i++; i++; r[Z] = A[i++];
}

static int read_matrix(FILE *f, double A[16]) {
    int i;
    for (i = 0; i < 16; i++)
        if (fscanf(f, "%lf", &A[i]) != 1) return 0;
    return 1;
}

void matrices_read(const char *path, /**/ Matrices **pq) {
    int n;
    FILE *f;
    Matrices *q;
    EMALLOC(1, &q);
    EMALLOC(MAX_N, &q->m);
    UC(efopen(path, "r", /**/ &f));
    UC(efclose(f));
    n = 0;
    while (read_matrix(f, /**/ q->m[n++].D))
        if (n > MAX_N) ERR("n=%d > MAX_N=%d", n, MAX_N);
    q->n = n;
    *pq = q;
}

static int good(const Coords *c, const double A[16]) {
    enum {X, Y, Z};
    double r[3];
    matrix2r(A, /**/ r);
    return
        xlo(c) < r[X] && r[X] <= xhi(c) &&
        ylo(c) < r[Y] && r[Y] <= yhi(c) &&
        zlo(c) < r[Z] && r[Z] <= zhi(c);
}
void matrices_read_filter(const char *path, const Coords *c, /**/ Matrices **pq) {
    int n;
    FILE *f;
    Matrices *q;
    EMALLOC(1, &q);
    EMALLOC(MAX_N, &q->m);
    UC(efopen(path, "r", /**/ &f));
    UC(efclose(f));
    n = 0;
    while (read_matrix(f, /**/ q->m[n].D)) {
        if (good(c, q->m[n].D)) n++;
        if (n > MAX_N) ERR("n=%d > MAX_N=%d", n, MAX_N);
    }
    q->n = n;
    *pq = q;
}

void matrices_get(Matrices *q, int i, /**/ double **pq) {
    int n;
    n = q->n;
    if (i >= n) ERR("i=%d >= n=%d", i, n);
    *pq = q->m[i].D;
}

void matrices_get_r(Matrices *q, int k, /**/ double r[3]) {
    enum {X, Y, Z};
    int n;
    double *A;
    n = q->n;
    if (k >= n) ERR("k=%d >= n=%d", k, n);
    A = q->m[k].D;
    matrix2r(A, /**/ r);
}

int matrices_get_n(Matrices *q) { return q->n; }
void matrices_fin(Matrices *q) { EFREE(q->m); EFREE(q); }
