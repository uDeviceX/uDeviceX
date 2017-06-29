void Distr::fin() {
    for(int i = 0; i < 27; ++i) {
        CC(cudaFree(s.iidx_[i]));
        if (i) {
            CC(cudaFreeHost(s.pp_hst_[i]));
            CC(cudaFreeHost(r.pp_hst_[i]));

            if (global_ids) {
                CC(cudaFreeHost(s.ii_hst_[i]));
                CC(cudaFreeHost(r.ii_hst_[i]));
            }
        } else {
            CC(                cudaFree(s.pp_hst_[i])); /* r.pp_hst_[0] = s.pp_hst_[0] */
            if (global_ids) CC(cudaFree(s.ii_hst_[i])); /* r.ii_hst_[0] = s.ii_hst_[0] */
        }
    }

    CC(cudaFree(s.iidx));
    CC(cudaFree(s.pp_dev)); CC(cudaFree(r.pp_dev));
    if (global_ids) {
        CC(cudaFree(s.ii_dev)); CC(cudaFree(r.ii_dev));
    }

    CC(cudaFree(s.size_dev)); CC(cudaFree(s.strt));
    CC(cudaFree(r.strt));

    delete s.size_pin;
}
