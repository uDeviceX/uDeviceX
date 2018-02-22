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
    double com[3] = {0.0, 0.0, 0.0};
    msg_print("nv: %d", nv);
    for (i = 0; i < nv; i++) {
        Positions_get(pos, i, /**/ r);
        com[X] += r[X];
        com[Y] += r[Y];
        com[Z] += r[Z];
    }
    com[X] /= nv; com[Y] /= nv; com[X] /= nv;
    msg_print("com: %g %g %g", com[X], com[Y], com[Z]);
}
float mesh_volume_apply0(MeshVolume *q, Positions *p) {
    UC(to_com(q->nv, p, /**/ q->rr));
    return 0.0;
}
