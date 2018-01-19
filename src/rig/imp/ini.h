static void load_rigid_mesh(const char *fname, int *nt, int *nv, int4 **tt_hst, int4 **tt_dev, float **vv_hst, float **vv_dev) {
    ply::read(fname, /**/ nt, nv, tt_hst, vv_hst);

    CC(d::Malloc((void**)tt_dev,     (*nt) * sizeof(int4)));
    CC(d::Malloc((void**)vv_dev, 3 * (*nv) * sizeof(float)));

    cH2D(*tt_dev, *tt_hst,     *nt);
    cH2D(*vv_dev, *vv_hst, 3 * *nv);
}


void ini(RigQuants *q) {
    q->n = q->ns = q->nps = 0;
    
    Dalloc(&q->pp ,     MAX_PART_NUM);
    Dalloc(&q->ss ,     MAX_SOLIDS);
    Dalloc(&q->rr0, 3 * MAX_PART_NUM);
    Dalloc(&q->i_pp,    MAX_PART_NUM);
    
    q->pp_hst   = new Particle[MAX_PART_NUM];
    q->ss_hst   = new Solid[MAX_SOLIDS];
    q->rr0_hst  = new float[3 * MAX_PART_NUM];
    q->i_pp_hst = new Particle[MAX_PART_NUM];

    q->ss_dmp    = new Solid[MAX_SOLIDS];
    q->ss_dmp_bb = new Solid[MAX_SOLIDS];

    load_rigid_mesh("mesh_solid.ply", /**/ &q->nt, &q->nv,
                    &q->htt, &q->dtt, &q->hvv, &q->dvv);
}
