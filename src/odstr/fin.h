void fin_S(Send *s) {
    for(int i = 0; i < 27; ++i) CC(cudaFree(s->iidx_[i]));
    CC(cudaFree(s->iidx));
    
    CC(cudaFree(s->pp.dp[0])); /* r.pp.dp[0] = s.pp.dp[0] */
    dealloc(&s->pp);
    
    if (global_ids) {
        CC(cudaFree(s->ii.dp[0])); /* r.ii.dp[0] = s.ii.dp[0] */
        dealloc(&s->ii);
    }

    CC(cudaFree(s->size_dev)); CC(cudaFree(s->strt));
    delete s->size_pin;
}

void fin_R(Recv *r) {
    dealloc(&r->pp);
    CC(cudaFree(r->strt));

    if (global_ids) dealloc(&r->ii);
}

void fin_SRI(Pbufs<int> *sii, Pbufs<int> *rii) {
    CC(cudaFree(sii->dp[0])); /* rii.dp[0] = sii.dp[0] */
    dealloc(sii);
    dealloc(rii);
}
