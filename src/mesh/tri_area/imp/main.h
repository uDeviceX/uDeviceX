void mesh_tri_area_ini(MeshRead *mesh, MeshTriArea **pq) {
    int nv, nt;
    MeshTriArea *q;
    EMALLOC(1, &q);
    nt = mesh_read_get_nt(mesh);
    nv = mesh_read_get_nv(mesh);
    EMALLOC(nt, &q->tt);

    q->nv = nv; q->nt = nt;
    EMEMCPY(nt, mesh_read_get_tri(mesh), q->tt);

    *pq = q;
}

void mesh_tri_area_fin(MeshTriArea *q) { EFREE(q->tt); EFREE(q); }
static void get(Positions *p, int i, double d[3]) {
    enum {X, Y, Z};
    float f[3];
    UC(positions_get(p, i, /**/ f));
    d[X] = f[X]; d[Y] = f[Y]; d[Z] = f[Z];
}
static double area(int nt, int4 *tt, Positions *p, int offset) {
    enum {X, Y, Z};
    int i, ia, ib, ic;
    double a[3], b[3], c[3], sum;
    KahanSum *kahan_sum;
    kahan_sum_ini(&kahan_sum);
    for (i = 0; i < nt; i++) {
        ia = tt[i].x; ib = tt[i].y; ic = tt[i].z;
        UC(get(p, ia + offset, /**/ a));
        UC(get(p, ib + offset, /**/ b));
        UC(get(p, ic + offset, /**/ c));
        kahan_sum_add(kahan_sum, tri_hst::kahan_area(a, b, c));
    }
    sum = kahan_sum_get(kahan_sum);
    return sum;
}
double mesh_tri_area_apply0(MeshTriArea *q, Positions *p) {
    int nt, offset;
    int4 *tt;
    nt = q->nt; tt = q->tt; offset = 0;
    return area(nt, tt, p, offset);
}

void mesh_tri_area_apply(MeshTriArea *q, int m, Positions *p, double *area0) {
    int i, nt, nv, offset;
    int4 *tt;
    nt = q->nt; tt = q->tt; nv = q->nv; offset = 0;

    for (i = 0; i < m; i++) {
        UC(area0[i] = area(nt, tt, p, offset));
        offset += nv;
    }
}
