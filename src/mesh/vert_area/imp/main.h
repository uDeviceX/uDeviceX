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
static void area(int4 t, Vectors *p, int offset, /**/ double *o) {
    int ia, ib, ic;
    double a[3], b[3], c[3], A;
    ia = t.x; ib = t.y; ic = t.z;
    UC(get(p, ia + offset, /**/ a));
    UC(get(p, ib + offset, /**/ b));
    UC(get(p, ic + offset, /**/ c));

    A = tri_hst::kahan_area(a, b, c)/3;
    o[ia + offset] += A;
    o[ib + offset] += A;
    o[ic + offset] += A;
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
