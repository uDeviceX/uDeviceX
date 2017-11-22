void pack(const Cloud *cloud, /**/ Pack *p) {
    int nc;
    int26 cc;
    int27 ss;
    Pap26 fpp;
    intp26 fcc;
    
    nc = get_cell_num(/**/ cc.d);
    scan(NFRAGS, cc.d, /**/ ss.d);

    bag2Sarray(p->dpp, /**/ &fpp);
    
    KL( dev::collect_particles,
        ((nc+1) / 2, 32),
        (ss, (const Particle*) cloud->pp, p->bss, p->bcc, p->fss, p->cap, /**/ p->bii, fpp, p->counts_dev));
    

    if (multi_solvent) {
        bag2Sarray(p->dcc, /**/ &fcc);

        KL( dev::collect_colors,
            ((nc+1) / 2, 32),
            (ss, cloud->cc, p->bss, p->bcc, p->fss, p->cap, /**/ fcc));        
    }
}

/* assume collect_[particles,colors] write to pinned memory */
void download_data(Pack *p) {
    int counts[26];
    d::MemcpyAsync(counts, p->counts_dev, sizeof(counts), D2H);
    dSync(); /* wait for collect_* and counts memcpy*/
}
