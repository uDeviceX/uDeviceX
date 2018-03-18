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

static void compute_com(int nv, Vectors *pos, /**/ double *com) {
    enum {X, Y, Z};
    int i;
    float r[3];
    KahanSum *sx, *sy, *sz;
    kahan_sum_ini(&sx); kahan_sum_ini(&sy); kahan_sum_ini(&sz);
    for (i = 0; i < nv; i++) {
        UC(positions_get(pos, i, /**/ r));
        kahan_sum_add(sx, r[X]);
        kahan_sum_add(sy, r[Y]);
        kahan_sum_add(sz, r[Z]);
    }
    com[X] = kahan_sum_get(sx)/nv;
    com[Y] = kahan_sum_get(sy)/nv;
    com[Z] = kahan_sum_get(sz)/nv;
    kahan_sum_fin(sx); kahan_sum_fin(sy); kahan_sum_fin(sz);
}
static void to_com(int nv, int offset, Vectors *pos, /**/ double *rr) {
    enum {X, Y, Z};
    int i;
    float   r0[3]; /* from */
    double *r1;    /* to */
    double com[3];
    UC(compute_com(nv, pos, /**/ com));
    for (i = 0; i < nv; i++) {
        UC(positions_get(pos, i + offset, /**/ r0));
        r1 = &rr[3*i];
        r1[X] = r0[X] - com[X];
        r1[Y] = r0[Y] - com[Y];
        r1[Z] = r0[Z] - com[Z];
    }
}
static double M0(double a[3], double b[3], double c[3]) {
    enum {X, Y, Z};
    return  (+a[X]*(b[Y]*c[Z]-b[Z]*c[Y])
             -a[Y]*(b[X]*c[Z]-b[Z]*c[X])
             +a[Z]*(b[X]*c[Y]-b[Y]*c[X]));
}
static void get(double *rr, int i, /**/ double *r1) {
    enum {X, Y, Z};
    double *r0;
    r0 = &rr[3*i];
    r1[X] = r0[X]; r1[Y] = r0[Y]; r1[Z] = r0[Z];
}
static double volume(int nt, int4 *tt, double *rr) {
    enum {X, Y, Z};
    int i, ia, ib, ic;
    double a[3], b[3], c[3], sum;
    KahanSum *kahan_sum;
    kahan_sum_ini(&kahan_sum);
    for (i = 0; i < nt; i++) {
        ia = tt[i].x; ib = tt[i].y; ic = tt[i].z;
        UC(get(rr, ia, /**/ a));
        UC(get(rr, ib, /**/ b));
        UC(get(rr, ic, /**/ c));
        kahan_sum_add(kahan_sum, M0(a, b, c));
    }
    sum = kahan_sum_get(kahan_sum);
    return sum/6;
}
double mesh_volume_apply0(MeshVolume *q, Vectors *positions) {
    int nv, nt, offset;
    double *rr;
    int4 *tt;
    nv = q->nv; nt = q->nt; rr = q->rr; tt = q->tt; offset = 0;
    UC(to_com(nv, offset, positions, /**/ rr));
    return volume(nt, tt, rr);
}

 void mesh_volume_apply(MeshVolume *q, int m, Vectors *positions, double *volume0) {
    int i, nt, nv, offset;
    double *rr;    
    int4 *tt;
    nv = q->nv; nt = q->nt; rr = q->rr; tt = q->tt; offset = 0;    
    for (i = 0; i < m; i++) {
        UC(to_com(nv, offset, positions, /**/ rr));
        UC(volume0[i] = volume(nt, tt, rr));
        offset += nv;
    }
}
