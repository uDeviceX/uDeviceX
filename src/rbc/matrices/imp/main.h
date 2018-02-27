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

static int read_r(FILE *f, double A[16]) {
    enum {X, Y, Z};
    int i;
    double r[3];
    for (i = 0; i < 3; i++)
        if (fscanf(f, "%lf", &r[i]) != 1) return 0;
    i = 0;
    A[i++] = 1; A[i++] = 0; A[i++] = 0; A[i++] = r[X];
    A[i++] = 0; A[i++] = 1; A[i++] = 0; A[i++] = r[Y];
    A[i++] = 0; A[i++] = 0; A[i++] = 1; A[i++] = r[Z];
    A[i++] = 0; A[i++] = 0; A[i++] = 0; A[i++] = 1;
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

void matrices_read_r(const char *path, /**/ Matrices **pq) {
    int n;
    FILE *f;
    Matrices *q;
    EMALLOC(1, &q);
    EMALLOC(MAX_N, &q->m);
    UC(efopen(path, "r", /**/ &f));
    UC(efclose(f));
    n = 0;
    while (read_r(f, /**/ q->m[n++].D)) ;
    q->n = n;
    *pq = q;
}

int good(Coords *c, const double A[16]) {
    double r[3];
    matrix2r(A, /**/ r);
    return 1;
}

void copy(const double A[16], double B[16]) {
    int i;
    for (i = 0; i < 16; i++) B[i] = A[i];
}

void matrices_ini_filter(Matrices *from, Coords *c, /**/ Matrices **pq) {
    int i, m, n;
    Matrices *to;
    EMALLOC(1, &to);
    EMALLOC(MAX_N, &to->m);
    n = from->n;
    for (i = m = 0; i < n; i++)
        if (good(c, from->m[i].D))
            copy(from->m[i].D, /**/ to->m[m++].D);
    to->n = m;
    *pq = to;
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
