
void pack(const Cloud *cloud, /**/ Pack *p) {
    int nc;
    int26 cc;
    int27 ss;
    Pap26 fpp;
    
    nc = get_cell_num(/**/ cc.d);
    scan(NFRAGS, cc.d, /**/ ss.d);

    bag2Sarray(p->dpp, /**/ &fpp);
    
    KL( dev::collect_particles,
        ((nc+1) / 2, 32),
        (ss, (const Particle*) cloud->pp, p->bss, p->bcc, p->fss, p->cap, /**/ p->bii, fpp, p->counts_dev));
    
    
}
