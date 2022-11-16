void mesh_angle_ini(MeshRead *mesh, MeshAngle **pq) {
    int i, nv, nd;
    MeshAngle *q;
    const int4 *dd;
    UC(nv = mesh_read_get_nv(mesh));
    UC(nd = mesh_read_get_ne(mesh));
    UC(dd = mesh_read_get_dih(mesh));
    EMALLOC(1, &q);
    EMALLOC(nd, &q->dd);
    for (i = 0; i < nd; i++)
        q->dd[i] = dd[i];
    q->nd = nd; q->nv = nv;
    *pq = q;
}

void mesh_angle_fin(MeshAngle *q) {
    EFREE(q->dd); EFREE(q);
}

static void get(Vectors *p, int i, double d[3]) {
    enum {X, Y, Z};
    float f[3];
    UC(vectors_get(p, i, /**/ f));
    d[X] = f[X]; d[Y] = f[Y]; d[Z] = f[Z];
}
static double angle(int4 t, Vectors *p, int offset) {
    int ia, ib, ic, id;
    double a[3], b[3], c[3], d[3], x, y;
    ia = t.x; ib = t.y; ic = t.z; id = t.w;

    UC(get(p, ia + offset, /**/ a));
    UC(get(p, ib + offset, /**/ b));
    UC(get(p, ic + offset, /**/ c));
    UC(get(p, id + offset, /**/ d));
    
    tri_hst::dihedral_xy(a, b, c, d, /**/ &x, &y);
    return atan2(y, x);
}
void mesh_angle_apply(MeshAngle *q, int m, Vectors *p, /**/ double *angle0) {
    int i, j, k, nv, nd, offset;
    int4 *dd;
    nd = q->nd; nv = q->nv; dd = q->dd; offset = 0;

    for (k = i = 0; i < m; i++) {
        for (j = 0; j < nd; j++)
            angle0[k++] = angle(dd[j], p, offset);
        offset += nv;
    }
}
