void rig_fin(RigQuants *q) {
    EFREE(q->pp_hst);
    EFREE(q->ss_hst);
    EFREE(q->rr0_hst);
    EFREE(q->i_pp_hst);
    
    Dfree(q->pp);
    Dfree(q->ss);
    Dfree(q->rr0);
    Dfree(q->i_pp);

    Dfree(q->dtt);
    Dfree(q->dvv);
    
    EFREE(q->ss_dmp);
}
