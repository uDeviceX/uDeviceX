static bool is_stress_free(const RbcForce *f) {
    return f->stype == RBC_SFREE;
}

void rbc_force_ini(const MeshRead *cell, int seed, RbcForce **pq) {
    RbcForce *q;
    int md, nt, nv;
    const int4 *tt;
    EMALLOC(1, &q);
    nv = mesh_get_nv(cell);
    nt = mesh_get_nt(cell);
    // md = mesh_get_md(cell);
    md = RBCmd;
    tt = mesh_get_tri(cell);
    
    UC(adj_ini(md, nt, nv, tt, /**/ &q->adj));
    UC(adj_view_ini(q->adj, /**/ &q->adj_v));

    if (RBC_RND) rbc_rnd_ini(nv*md*MAX_CELL_NUM, seed, &q->rnd);
    *pq = q;
}

static void fin_rnd(RbcRnd *rnd) {
    rbc_rnd_fin(rnd);
}

static void fin_stress(RbcForce *f) {
    if (is_stress_free(f)) {
        StressFree_v v = f->sinfo.sfree;
        Dfree(v.ll);
        Dfree(v.aa);
    }
}

void rbc_force_fin(RbcForce *q) {
    if (RBC_RND) fin_rnd(q->rnd);
    UC(fin_stress(q));
    UC(adj_fin(q->adj));
    UC(adj_view_fin(q->adj_v));
    EFREE(q);
}

void rbc_force_set_stressful(int nt, float totArea, /**/ RbcForce *f) {
    StressFul_v v;
    float a0 = totArea / nt;

    v.a0 = a0;
    v.l0 = sqrt(a0 * 4.0 / sqrt(3.0));
    
    f->stype = RBC_SFUL;
    f->sinfo.sful = v;
}

void rbc_force_set_stressfree(const char *fname, /**/ RbcForce *f) {
    StressFree_v v;
    MeshRead *cell;
    Adj *adj;
    RbcShape *shape;
    const float *rr;
    float *ll_hst, *aa_hst;
    const int4 *tt;
    int n, nv, nt, md;
    
    UC(mesh_read_off(fname, &cell));
    rr = mesh_get_vert(cell);
    tt = mesh_get_tri(cell);
    nt = mesh_get_nt(cell);
    nv = mesh_get_nv(cell);
    md = mesh_get_md(cell);

    UC(adj_ini(md, nt, nv, tt, /**/ &adj));
    UC(rbc_shape_ini(adj, rr, /**/ &shape));

    n = adj_get_max(adj);

    rbc_shape_edg(shape, &ll_hst);
    rbc_shape_area(shape, &aa_hst);

    Dalloc(&v.ll, n);    
    Dalloc(&v.aa, n);
    
    cH2D(v.ll, ll_hst, n);
    cH2D(v.aa, aa_hst, n);
    
    UC(rbc_shape_fin(shape));
    UC(adj_fin(adj));
    UC(mesh_fin(cell));

    f->stype = RBC_SFREE;
    f->sinfo.sfree = v;
}
