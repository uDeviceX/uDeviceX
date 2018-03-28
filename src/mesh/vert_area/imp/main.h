void mesh_vert_area_ini(MeshRead *mesh, MeshVertArea **pq) {
    int nv, nt;
    MeshVertArea *q;
    EMALLOC(1, &q);
    nt = mesh_read_get_nt(mesh);
    nv = mesh_read_get_nv(mesh);
    EMALLOC(nt, &q->tt);

    q->nv = nv; q->nt = nt;
    EMEMCPY(nt, mesh_read_get_tri(mesh), q->tt);

    *pq = q;
}

void mesh_vert_area_fin(MeshVertArea *q) { EFREE(q->tt); EFREE(q); }
static void get(Vectors *p, int i, double d[3]) {
    enum {X, Y, Z};
    float f[3];
    UC(vectors_get(p, i, /**/ f));
    d[X] = f[X]; d[Y] = f[Y]; d[Z] = f[Z];
}
static void mid(double a[3], double b[3], /**/ double c[3]) {
    enum {X, Y, Z};
    c[X] = (a[X] + b[X])/2;
    c[Y] = (a[Y] + b[Y])/2;
    c[Z] = (a[Z] + b[Z])/2;
}
static double area0(double a[3], double b[3], double c[3]) {
    double ans;
    ans = tri_hst::kahan_area(a, b, c);
    if (ans <= 0) ERR("area=%g <= 0", ans);
    return ans;
}
static void area(int4 t, Vectors *p, int offset, /**/ double *o) {
    int ia, ib, ic;
    double a[3], b[3], c[3], ab[3], ac[3], bc[3];
    ia = t.x; ib = t.y; ic = t.z;
    UC(get(p, ia + offset, /**/ a));
    UC(get(p, ib + offset, /**/ b));
    UC(get(p, ic + offset, /**/ c));

    mid(a, b, /**/ ab);
    mid(a, c, /**/ ac);
    mid(b, c, /**/ bc);

    o[ia + offset] += area0(a, ab, ac);
    o[ib + offset] += area0(b, bc, ab);
    o[ic + offset] += area0(c, ac, bc);
}

void mesh_vert_area_apply(MeshVertArea *q, int nm, Vectors *pos, double *o) {
    int i, j, nt, nv, n, offset;
    int4 *tt;
    nt = q->nt; tt = q->tt; nv = q->nv; offset = 0;
    n = nv * nm;
    for (i = 0; i < n; i++) o[i] = 0;

    for (i = 0; i < nm; i++) {
        for (j = 0; j < nt; j++)
            area(tt[j], pos, offset, /**/ o);
        offset += nv;
    }
}
