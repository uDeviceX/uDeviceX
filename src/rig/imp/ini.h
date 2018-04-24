void rig_ini(long maxs, long maxp, const MeshRead *mesh, RigQuants *q) {
    q->n = q->ns = q->nps = 0;
    q->maxp = maxp;

    Dalloc(&q->pp,      maxp);
    Dalloc(&q->ss,      maxs);
    Dalloc(&q->rr0, 3 * maxp);
    Dalloc(&q->i_pp,    maxp);

    EMALLOC(maxp, &q->pp_hst);
    EMALLOC(maxs, &q->ss_hst);
    EMALLOC(3 * maxp, &q->rr0_hst);
    EMALLOC(maxp, &q->i_pp_hst);

    EMALLOC(maxs, &q->ss_dmp);
    EMALLOC(maxs, &q->ss_dmp_bb);

    q->nt = mesh_read_get_nt(mesh);
    q->nv = mesh_read_get_nv(mesh);
    
    Dalloc(&q->dtt,     q->nt);
    Dalloc(&q->dvv, 3 * q->nv);

    cH2D(q->dtt, mesh_read_get_tri (mesh),     q->nt);
    cH2D(q->dvv, mesh_read_get_vert(mesh), 3 * q->nv);

    //UC(load_rigid_mesh("rig.ply", /**/ &q->nt, &q->nv, &q->htt, &q->dtt, &q->hvv, &q->dvv));
}
