void rig_fin(RigQuants *q) {
    delete[] q->pp_hst;
    delete[] q->ss_hst;
    delete[] q->rr0_hst;
    delete[] q->i_pp_hst;
    
    Dfree(q->pp);
    Dfree(q->ss);
    Dfree(q->rr0);
    Dfree(q->i_pp);

    if (q->htt) delete[] q->htt;
    if (q->hvv) delete[] q->hvv;

    if (q->dtt) CC(d::Free(q->dtt));
    if (q->dvv) CC(d::Free(q->dvv));

    delete[] q->ss_dmp;
    delete[] q->ss_dmp_bb;
}
