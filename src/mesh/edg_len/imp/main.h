void mesh_edg_len_ini(MeshRead *mesh, MeshEdgLen **pq) {
    MeshEdgLen *q;
    int i, j, nv, ne, *ee;
    const int4 *dd;
    int4 d;
    EMALLOC(1, &q);
    nv = mesh_read_get_nv(mesh);
    ne = mesh_read_get_ne(mesh);
    dd = mesh_read_get_dih(mesh);
    EMALLOC(2*ne, &ee);
    for (j = i = 0; i < ne; i++) {
        d = dd[i];
        ee[j++] = d.y; ee[j++] = d.z;
    }
    q->nv = nv; q->ne = ne; q->ee = ee;
    *pq = q;
}

static void get(Vectors *p, int i, double d[3]) {
    enum {X, Y, Z};
    float f[3];
    UC(vectors_get(p, i, /**/ f));
    d[X] = f[X]; d[Y] = f[Y]; d[Z] = f[Z];
}
static double sum_sq(double x, double y, double z) { return x*x + y*y + z*z; }
static double len0(Vectors *pos, int i, int j) {
    enum {X, Y, Z};
    double d, a[3], b[3];
    get(pos, i, /**/ a);
    get(pos, j, /**/ b);
    d = sum_sq(a[X] - b[X], a[Y] - b[Y], a[Z] - b[Z]);
    return sqrt(d);
}
static void len(MeshEdgLen *q, Vectors *pos, int offset_v, int offset_e, /**/ double *o) {
    int ne, i, k, a, b;
    double value;
    const int *ee;
    ne = q->ne;
    ee = q->ee;
    for (k = i = 0; i < ne; i++) {
        a = ee[k++]; b = ee[k++];
        UC(value = len0(pos, a + offset_v, b + offset_v));
        o[a + offset_e] += value;
    }
}
void mesh_edg_len_apply(MeshEdgLen *q, int nm, Vectors *pos, /**/ double *o) {
    int i, nv, ne;
    int offset_v, offset_e; /* vert and edge */
    nv = q->nv;
    ne = q->ne;
    for (offset_e = offset_v = i = 0; i < nm; i++) {
        UC(len(q, pos, offset_v, offset_e, /**/ o));
        offset_v += nv;
        offset_e += ne;
    }
}

void mesh_edg_len_fin(MeshEdgLen *q) {
    EFREE(q->ee);
    EFREE(q);
}
