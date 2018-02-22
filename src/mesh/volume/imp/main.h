void mesh_volume_ini(MeshRead *mesh, MeshVolume **pq) {
    int nv, nt;
    MeshVolume *q;
    EMALLOC(1, &q);
    nv = mesh_read_get_nv(mesh);
    nt = mesh_read_get_nt(mesh);
    EMALLOC(3*nv, &q->rr);
    EMALLOC(  nt, &q->tt);

    q->nv = nv;
    q->nt = nt;
    EMEMCPY(nt, mesh_read_get_tri(mesh), q->tt);
    
    *pq = q;
}

void mesh_volume_fin(MeshVolume *q) {
    EFREE(q->rr);
    EFREE(q->tt);
    EFREE(q);
}

static void compute_com(int nv, Positions *pos, /**/ double *com) {
    enum {X, Y, Z};
    int i;
    float r[3];
    KahanSum *sx, *sy, *sz;
    kahan_sum_ini(&sx); kahan_sum_ini(&sy); kahan_sum_ini(&sz);
    for (i = 0; i < nv; i++) {
        UC(Positions_get(pos, i, /**/ r));
        kahan_sum_add(sx, r[X]);
        kahan_sum_add(sy, r[Y]);
        kahan_sum_add(sz, r[Z]);
    }
    com[X] = kahan_sum_get(sx)/nv;
    com[Y] = kahan_sum_get(sy)/nv;
    com[Z] = kahan_sum_get(sz)/nv;
    kahan_sum_fin(sx); kahan_sum_fin(sy); kahan_sum_fin(sz);
}
static void to_com(int nv, int offset, Positions *pos, /**/ double *rr) {
    enum {X, Y, Z};
    int i;
    float   r0[3]; /* from */
    double *r1;    /* to */
    double com[3];
    UC(compute_com(nv, pos, /**/ com));
    for (i = 0; i < nv; i++) {
        UC(Positions_get(pos, i + offset, /**/ r0));
        r1 = &rr[3*i];
        r1[X] = r0[X] - com[X];
        r1[Y] = r0[Y] - com[Y];
        r1[Z] = r0[Z] - com[Z];
    }
}
// static void M0(const float *A, const float *B, const float *C, /**/ float *res) {
//     *res =  (+Ax * (By * Cz -Bz * Cy)
//              -Ay * (Bx * Cz -Bz * Cx)
//              +Az * (Bx * Cy -By * Cx)) / 6.f;
// }
static void get(double *rr, int i, double *r1) {
    enum {X, Y, Z};
    double *r0;
    r0 = &rr[3*i];
    r1[X] = r0[X]; r1[Y] = r0[Y]; r1[Z] = r0[Z];
}
static double volume(int nt, int4 *tt, double *rr) {
    enum {X, Y, Z};
    int i, ia, ib, ic;
    double sum;
    double a[3], b[3], c[3];
    
    sum = 0;
    for (i = 0; i < nt; i++) {
        ia = tt[i].x; ib = tt[i].y; ic = tt[i].z;
        UC(get(rr, ia, /**/ a));
        UC(get(rr, ib, /**/ b));
        UC(get(rr, ic, /**/ c));
    }
    return sum/nt;
}
float mesh_volume_apply0(MeshVolume *q, Positions *p) {
    int nv, nt, offset;
    double *rr;
    int4 *tt;
    nv = q->nv; nt = q->nt; rr = q->rr; tt = q->tt; offset = 0;
    UC(to_com(nv, offset, p, /**/ rr));
    return volume(nt, tt, rr);
}
