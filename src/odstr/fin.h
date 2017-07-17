void Distr::fin() {
    for(int i = 0; i < 27; ++i) {
        CC(cudaFree(s.iidx_[i]));
        if (i) {
            if (global_ids) {
                CC(cudaFreeHost(r.ii_hst_[i]));
            }
        } else {
            CC(                cudaFree(s.pp.dp[i])); /* r.pp_hst_[0] = s.pp_hst_[0] */
            if (global_ids) CC(cudaFree(s.ii.dp[i])); /* r.ii_hst_[0] = s.ii_hst_[0] */
        }
    }

    dealloc(&s.pp);
    dealloc(&r.pp);
    
    CC(cudaFree(s.iidx));

    if (global_ids) {
        dealloc(&s.ii);
        CC(cudaFree(r.ii_dev));
    }

    CC(cudaFree(s.size_dev)); CC(cudaFree(s.strt));
    CC(cudaFree(r.strt));

    delete s.size_pin;
}
