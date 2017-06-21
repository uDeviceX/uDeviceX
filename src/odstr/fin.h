void Fluid::fin() {
    for(int i = 0; i < 27; ++i) {
        CC(cudaFree(s.iidx_[i]));
        if (i) {
            CC(cudaFreeHost(s.hst_[i]));
            CC(cudaFreeHost(r.hst_[i]));
        } else {
            CC(cudaFree    (s.hst_[i])); /* r.hst_[0] = s.hst_[0] */
        }
    }

    CC(cudaFree(s.iidx));
    CC(cudaFree(s.dev)); CC(cudaFree(r.dev));

    CC(cudaFree(s.size_dev)); CC(cudaFree(s.strt));
    CC(cudaFree(r.strt)); CC(cudaFree(r.strt_pa));

    delete s.size_pin;
}
