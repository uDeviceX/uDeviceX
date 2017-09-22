void forces_cnt(std::vector<PaWrap> *pwr, std::vector<FoWrap> *fwr) {
    cnt::bind(*pwr, *fwr);
    cnt::bulk(*pwr, *fwr);
}

void forces_fsi(fsi::SolventWrap *w_s, std::vector<PaWrap> *pwr, std::vector<FoWrap> *fwr) {
    fsi::bind(*w_s);
    fsi::bulk(*pwr, *fwr);
}

void forces_objects() {
    fsi::SolventWrap w_s;
    hforces::Cloud cloud;
    std::vector<PaWrap> pwr;
    std::vector<FoWrap> fwr;

    if (solids0) {
        pwr.push_back({s::q.n, s::q.pp});
        fwr.push_back({s::q.n, s::ff});
    }
    if (rbcs) {
        pwr.push_back({r::q.n, r::q.pp});
        fwr.push_back({r::q.n, r::ff});
    }

    if (contactforces) forces_cnt(&pwr, &fwr);

    hforces::ini_cloud(o::q.pp, &cloud);
    if (multi_solvent) hforces::ini_cloud_color(o::qc.ii, &cloud);
    w_s.pp = o::q.pp;
    w_s.c  = cloud;
    w_s.ff = o::ff;
    w_s.n  = o::q.n;
    w_s.starts = o::q.cells.starts;
    if (fsiforces)     forces_fsi(&w_s, &pwr, &fwr);

    /* temporary */
    std::vector<ParticlesWrap> w_r;
    if (solids0) w_r.push_back(ParticlesWrap(s::q.pp, s::q.n, s::ff));
    if (rbcs)    w_r.push_back(ParticlesWrap(r::q.pp, r::q.n, r::ff));
    rex::rex(w_r); /* fsi::halo(), cnt::halo() */
}

void forces_objects_new() {
    fsi::SolventWrap w_s;
    hforces::Cloud cloud;
    std::vector<PaWrap> pwr;
    std::vector<FoWrap> fwr;

    if (solids0) {
        pwr.push_back({s::q.n, s::q.pp});
        fwr.push_back({s::q.n, s::ff});
    }
    if (rbcs) {
        pwr.push_back({r::q.n, r::q.pp});
        fwr.push_back({r::q.n, r::ff});
    }
    // TODO rm std::vector
    int nw = pwr.size();

    /* Prepare and send the data */
    
    using namespace rs;
    build_map(nw, pwr.data(), /**/ &e.p);
    pack(nw, pwr.data(), /**/ &e.p);
    download(nw, /**/ &e.p);

    post_send(&e.p, &e.c);
    post_recv(&e.c, &e.u);

    /* bulk interactions */
    
    hforces::ini_cloud(o::q.pp, &cloud);
    if (multi_solvent) hforces::ini_cloud_color(o::qc.ii, &cloud);

    w_s.pp = o::q.pp;
    w_s.c  = cloud;
    w_s.ff = o::ff;
    w_s.n  = o::q.n;
    w_s.starts = o::q.cells.starts;

    if (contactforces) forces_cnt(&pwr, &fwr);
    if (fsiforces)     forces_fsi(&w_s, &pwr, &fwr);

    /* recv data and halo interactions  */

    wait_send(&e.c);
    wait_recv(&e.c, &e.u);

    int26 hcc = get_counts(&e.u);
    Pap26 hpp = upload_shift(&e.u);
    Fop26 hff = reini_ff(&e.p, &e.pf);

    if (fsiforces)     fsi::halo(hpp, hff, hcc.d);
    if (contactforces) cnt::halo(hpp, hff, hcc.d);

    /* send the forces back */ 
    
    download_ff(&e.pf);

    post_send_ff(&e.pf, &e.c);
    post_recv_ff(&e.c, &e.uf);

    wait_send_ff(&e.c);    
    wait_recv_ff(&e.c, &e.uf);

    unpack_ff(&e.uf, &e.p, nw, /**/ fwr.data());
}

