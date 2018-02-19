static void ini(OffRead **pq) {
    OffRead *p;
    UC(emalloc(sizeof(OffRead), (void**)&p));
    *pq = p;
}

void off_read_off(const char *path, OffRead **pq) {
    FILE *f;
    OffRead *q;
    UC(ini(&q));
    UC(efopen(path, "r", /**/ &f));
    read_off(f, path, /**/ q);
    UC(efclose(f));
    msg_print("read '%s'", path);
    *pq = q;
}

void off_fin(OffRead* q) { EFREE(q->rr); EFREE(q->tt); EFREE(q); }
int off_get_nv(OffRead *q) { return q->nv; }
int off_get_nt(OffRead *q) { return q->nt; }
static int amax(int *a, int n) {
    int i, m;
    if (n <= 0) ERR("amax called with size: %d\n", n);
    m = a[0];
    for (i = 1; i < n; i++)
        if (a[i] > m) m = a[i];
    return m;
}
int off_get_md(OffRead *q) {
    int *d;
    int i, m, nt, nv;
    int x, y, z;
    int4 *tt;
    nt = q->nt; nv = q->nv; tt = q->tt;

    if (nv == 0) {
        msg_print("off_get_md called for nv = 0");
        return 0;
    }
    UC(emalloc(nv*sizeof(d[0]), (void**)&d));
    for (i = 0; i < nv; i++) d[i] = 0;
    for (i = 0; i < nt; i++) {
        x = tt[i].x; y = tt[i].y; z = tt[i].z;
        d[x]++; d[y]++; d[z]++;
    }
    m = amax(d, nv);
    UC(efree(d));
    return m;
}

const float *off_get_vert(OffRead *q) { return q->rr; }
const int4  *off_get_tri(OffRead *q) { return q->tt; }
