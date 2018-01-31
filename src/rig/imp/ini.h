static void load_rigid_mesh(const char *fname, int *nt, int *nv, int4 **tt_hst, int4 **tt_dev, float **vv_hst, float **vv_dev) {
    msg_print("reading: '%s'", fname);
    UC(ply::read(fname, /**/ nt, nv, tt_hst, vv_hst));

    CC(d::Malloc((void**)tt_dev,     (*nt) * sizeof(int4)));
    CC(d::Malloc((void**)vv_dev, 3 * (*nv) * sizeof(float)));

    cH2D(*tt_dev, *tt_hst,     *nt);
    cH2D(*vv_dev, *vv_hst, 3 * *nv);
}


void rig_ini(int maxp, RigQuants *q) {
    q->n = q->ns = q->nps = 0;
    q->maxp = maxp;

    Dalloc(&q->pp ,     maxp);
    Dalloc(&q->ss ,     MAX_SOLIDS);
    Dalloc(&q->rr0, 3 * maxp);
    Dalloc(&q->i_pp,    maxp);

    EMALLOC(maxp, &q->pp_hst);
    EMALLOC(MAX_SOLIDS, &q->ss_hst);
    EMALLOC(3 * maxp, &q->rr0_hst);
    EMALLOC(maxp, &q->i_pp_hst);

    EMALLOC(MAX_SOLIDS, &q->ss_dmp);
    EMALLOC(MAX_SOLIDS, &q->ss_dmp_bb);

    UC(load_rigid_mesh("rig.ply", /**/ &q->nt, &q->nv, &q->htt, &q->dtt, &q->hvv, &q->dvv));
}
