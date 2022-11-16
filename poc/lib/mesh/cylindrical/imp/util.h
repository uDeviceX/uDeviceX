static void compute_com(int nv, Vectors *pos, /**/ double *com) {
    enum {X, Y, Z};
    int i;
    float r[3];
    KahanSum *sx, *sy, *sz;
    kahan_sum_ini(&sx); kahan_sum_ini(&sy); kahan_sum_ini(&sz);
    for (i = 0; i < nv; i++) {
        UC(vectors_get(pos, i, /**/ r));
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
        UC(vectors_get(pos, i + offset, /**/ r0));
        r1 = &rr[3*i];
        r1[X] = r0[X] - com[X];
        r1[Y] = r0[Y] - com[Y];
        r1[Z] = r0[Z] - com[Z];
    }
}
