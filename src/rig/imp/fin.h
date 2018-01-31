void rig_fin(RigQuants *q) {
    EFREE(q->pp_hst);
    EFREE(q->ss_hst);
    EFREE(q->rr0_hst);
    EFREE(q->i_pp_hst);
    
    Dfree(q->pp);
    Dfree(q->ss);
    Dfree(q->rr0);
    Dfree(q->i_pp);

    if (q->htt) EFREE(q->htt);
    if (q->hvv) EFREE(q->hvv);

    if (q->dtt) CC(d::Free(q->dtt));
    if (q->dvv) CC(d::Free(q->dvv));
    
    EFREE(q->ss_dmp);
    EFREE(q->ss_dmp_bb);
}
