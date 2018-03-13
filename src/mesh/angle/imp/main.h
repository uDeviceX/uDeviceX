void mesh_angle_ini(MeshRead *mesh, MeshAngle **pq) {
    int nv, nt, nd;
    MeshAngle *q;
    UC(nt = mesh_read_get_nt(mesh));
    UC(nv = mesh_read_get_nv(mesh));
    UC(nd = mesh_read_get_ne(mesh));
    EMALLOC(1, &q);
    EMALLOC(nt, &q->tt);
    EMALLOC(nd, &q->dd);

    q->nv = nv; q->nt = nt; q->nd = nd;
    EMEMCPY(nt, mesh_read_get_tri(mesh), q->tt);

    *pq = q;
}

void mesh_angle_fin(MeshAngle *q) {
    EFREE(q->dd); EFREE(q->tt); EFREE(q);
}
static void get(Positions *p, int i, double d[3]) {
    enum {X, Y, Z};
    float f[3];
    UC(positions_get(p, i, /**/ f));
    d[X] = f[X]; d[Y] = f[Y]; d[Z] = f[Z];
}
static double area(int4 t, Positions *p, int offset) {
    int ia, ib, ic;
    double a[3], b[3], c[3];
    ia = t.x; ib = t.y; ic = t.z;
    UC(get(p, ia + offset, /**/ a));
    UC(get(p, ib + offset, /**/ b));
    UC(get(p, ic + offset, /**/ c));
    return tri_hst::kahan_area(a, b, c);
}

void mesh_angle_apply(MeshAngle *q, int m, Positions *p, double *area0) {
    int i, j, k, nt, nv, offset;
    int4 *tt;
    nt = q->nt; tt = q->tt; nv = q->nv; offset = 0;

    for (k = i = 0; i < m; i++) {
        for (j = 0; j < nt; j++)
            area0[k++] = area(tt[j], p, offset);
        offset += nv;
    }
}
