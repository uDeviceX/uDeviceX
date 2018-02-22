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

static void to_com(int nv, Positions *pos, /**/ float *rr) {
    enum {X, Y, Z};
    int i;
    float r[3];
    double com[3];
    KahanSum *sx, *sy, *sz;
    kahan_sum_ini(&sx); kahan_sum_ini(&sy); kahan_sum_ini(&sz);
    for (i = 0; i < nv; i++) {
        Positions_get(pos, i, /**/ r);
        kahan_sum_add(sx, r[X]);
        kahan_sum_add(sy, r[Y]);
        kahan_sum_add(sz, r[Z]);
    }
    com[X] = kahan_sum_get(sx)/nv;
    com[Y] = kahan_sum_get(sy)/nv;
    com[Z] = kahan_sum_get(sz)/nv;
    msg_print("com: %g %g %g", com[X], com[Y], com[Z]);
    kahan_sum_fin(sx); kahan_sum_fin(sy); kahan_sum_fin(sz);
}
float mesh_volume_apply0(MeshVolume *q, Positions *p) {
    UC(to_com(q->nv, p, /**/ q->rr));
    return 0.0;
}
