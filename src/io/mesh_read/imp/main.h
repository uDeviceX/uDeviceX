static void ini(MeshRead **pq) {
    MeshRead *p;
    UC(emalloc(sizeof(MeshRead), (void**)&p));
    *pq = p;
}

void mesh_read_ini_off(const char *path, MeshRead **pq) {
    FILE *f;
    MeshRead *q;
    UC(ini(&q));
    UC(efopen(path, "r", /**/ &f));
    read_off(f, path, /**/ q);
    UC(efclose(f));
    msg_print("read off '%s'", path);
    *pq = q;
}

void mesh_read_ini_ply(const char *path, MeshRead **pq) {
    FILE *f;
    MeshRead *q;
    UC(ini(&q));
    UC(efopen(path, "r", /**/ &f));
    read_ply(f, path, /**/ q);
    UC(efclose(f));
    msg_print("read ply '%s'", path);
    *pq = q;
}

void mesh_read_fin(MeshRead* q) { EFREE(q->rr); EFREE(q->tt); EFREE(q); }
int mesh_read_get_nv(const MeshRead *q) { return q->nv; }
int mesh_read_get_nt(const MeshRead *q) { return q->nt; }
static int amax(int *a, int n) {
    int i, m;
    if (n <= 0) ERR("amax called with size: %d\n", n);
    m = a[0];
    for (i = 1; i < n; i++)
        if (a[i] > m) m = a[i];
    return m;
}
int mesh_read_get_md(const MeshRead *q) {
    int *d;
    int i, m, nt, nv;
    int x, y, z;
    int4 *tt;
    nt = q->nt; nv = q->nv; tt = q->tt;

    if (nv == 0) {
        msg_print("mesh_get_md called for nv = 0");
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

const float *mesh_read_get_vert(const MeshRead *q) { return q->rr; }
const int4  *mesh_read_get_tri(const MeshRead *q) { return q->tt; }
