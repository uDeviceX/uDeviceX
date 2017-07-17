void Distr::fin() {
    for(int i = 0; i < 27; ++i) CC(cudaFree(s.iidx_[i]));

    CC(                cudaFree(s.pp.dp[0])); /* r.pp.dp[0] = s.pp.dp[0] */
    if (global_ids) CC(cudaFree(s.ii.dp[0])); /* r.ii.dp[0] = s.ii.dp[0] */
    
    dealloc(&s.pp);
    dealloc(&r.pp);
    
    CC(cudaFree(s.iidx));

    if (global_ids) {
        dealloc(&s.ii);
        dealloc(&r.ii);
    }

    CC(cudaFree(s.size_dev)); CC(cudaFree(s.strt));
    CC(cudaFree(r.strt));

    delete s.size_pin;
}
