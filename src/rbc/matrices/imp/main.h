static const int MAX_N = 100000;

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
    msg_print("read '%s'", path);
    UC(efopen(path, "r", /**/ &f));
    n = 0;
    while (read_matrix(f, /**/ q->m[n].D)) {
        n++;
        if (n > MAX_N) ERR("n=%d > MAX_N=%d", n, MAX_N);
    }
    UC(efclose(f));
    q->n = n;
    *pq = q;
}

static int good(const Coords *c, const double A[16]) {
    enum {X, Y, Z};
    double r[3];
    matrix2r(A, /**/ r);
    return
        xlo(c) <= r[X] && r[X] < xhi(c) &&
        ylo(c) <= r[Y] && r[Y] < yhi(c) &&
        zlo(c) <= r[Z] && r[Z] < zhi(c);
}
void matrices_read_filter(const char *path, const Coords *c, /**/ Matrices **pq) {
    int m, n;
    FILE *f;
    Matrices *q;
    EMALLOC(1, &q);
    EMALLOC(MAX_N, &q->m);
    msg_print("read '%s'", path);
    UC(efopen(path, "r", /**/ &f));
    n = m = 0;
    while (read_matrix(f, /**/ q->m[n].D)) {
        m++;
        if (good(c, q->m[n].D)) n++;
        if (n > MAX_N) ERR("n=%d > MAX_N=%d", n, MAX_N);
    }
    msg_print("took %d/%d matrices", n, m);
    UC(efclose(f));
    q->n = n;
    *pq = q;
}

void matrices_get(const Matrices *q, int i, /**/ double **pq) {
    int n;
    n = q->n;
    if (i >= n) ERR("i=%d >= n=%d", i, n);
    *pq = q->m[i].D;
}

void matrices_get_r(const Matrices *q, int k, /**/ double r[3]) {
    enum {X, Y, Z};
    int n;
    double *A;
    n = q->n;
    if (k >= n) ERR("k=%d >= n=%d", k, n);
    A = q->m[k].D;
    matrix2r(A, /**/ r);
}

int matrices_get_n(const Matrices *q) { return q->n; }

static void log(double A[16]) {
    int i, j;
    double a, b, c, d;
    for (i = j = 0; j < 4; j++) {
        a = A[i++]; b = A[i++]; c = A[i++]; d = A[i++];
        msg_print("%g %g %g %g", a, b, c, d);
    }
}
void matrices_log(const Matrices *q) {
    int n, i;
    n = q->n;
    msg_print("<matrices_log");
    for (i = 0; i < n; i++)
        if (i > 0) msg_print("");
    msg_print(">matrices_log");
}

void matrices_fin(Matrices *q) { EFREE(q->m); EFREE(q); }
