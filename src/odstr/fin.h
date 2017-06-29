void Distr::fin() {
    for(int i = 0; i < 27; ++i) {
        CC(cudaFree(s.iidx_[i]));
        if (i) {
            CC(cudaFreeHost(s.pp_hst_[i]));
            CC(cudaFreeHost(r.hst_[i]));
        } else {
            CC(cudaFree    (s.pp_hst_[i])); /* r.pp_hst_[0] = s.hst_[0] */
        }
    }

    CC(cudaFree(s.iidx));
    CC(cudaFree(s.pp_dev)); CC(cudaFree(r.dev));

    CC(cudaFree(s.size_dev)); CC(cudaFree(s.strt));
    CC(cudaFree(r.strt));

    delete s.size_pin;
}
