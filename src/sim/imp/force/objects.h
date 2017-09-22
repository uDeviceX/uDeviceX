void forces_cnt(int nw, PaWrap *pw, FoWrap *fw) {
    cnt::bind(nw, pw, fw);
    cnt::bulk(nw, pw, fw);
}

void forces_fsi(fsi::SolventWrap *w_s, int nw, PaWrap *pw, FoWrap *fw) {
    fsi::bind(*w_s);
    fsi::bulk(nw, pw, fw);
}

void forces_objects() {
    fsi::SolventWrap w_s;
    hforces::Cloud cloud;
    PaWrap pw[MAX_OBJ_TYPES];
    FoWrap fw[MAX_OBJ_TYPES];
    int nw = 0;
    
    if (solids0) {
        pw[nw] = {s::q.n, s::q.pp};
        fw[nw] = {s::q.n, s::ff};
        ++nw;
    }
    if (rbcs) {
        pw[nw] = {r::q.n, r::q.pp};
        fw[nw] = {r::q.n, r::ff};
        ++nw;
    }

    if (!nw) return;

    if (contactforces) forces_cnt(nw, pw, fw);

    hforces::ini_cloud(o::q.pp, &cloud);
    if (multi_solvent) hforces::ini_cloud_color(o::qc.ii, &cloud);
    w_s.pp = o::q.pp;
    w_s.c  = cloud;
    w_s.ff = o::ff;
    w_s.n  = o::q.n;
    w_s.starts = o::q.cells.starts;
    if (fsiforces)     forces_fsi(&w_s, nw, pw, fw);

    /* temporary */
    std::vector<ParticlesWrap> w_r;
    if (solids0) w_r.push_back(ParticlesWrap(s::q.pp, s::q.n, s::ff));
    if (rbcs)    w_r.push_back(ParticlesWrap(r::q.pp, r::q.n, r::ff));
    rex::rex(w_r); /* fsi::halo(), cnt::halo() */
}

void forces_objects_new() {
    fsi::SolventWrap w_s;
    hforces::Cloud cloud;
    PaWrap pw[MAX_OBJ_TYPES];
    FoWrap fw[MAX_OBJ_TYPES];
    int nw = 0;
    
    if (solids0) {
        pw[nw] = {s::q.n, s::q.pp};
        fw[nw] = {s::q.n, s::ff};
        ++nw;
    }
    if (rbcs) {
        pw[nw] = {r::q.n, r::q.pp};
        fw[nw] = {r::q.n, r::ff};
        ++nw;
    }

    if (!nw) return;

    /* Prepare and send the data */
    
    using namespace rs;
    build_map(nw, pw, /**/ &e.p);
    pack(nw, pw, /**/ &e.p);
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

    if (contactforces) forces_cnt(nw, pw, fw);
    if (fsiforces)     forces_fsi(&w_s, nw, pw, fw);

    /* recv data and halo interactions  */

    wait_send(&e.c);
    wait_recv(&e.c, &e.u);

    int26 hcc = get_counts(&e.u);
    Pap26 hpp = upload_shift(&e.u);
    Fop26 hff = reini_ff(&e.u, &e.pf);

    if (fsiforces)     fsi::halo(hpp, hff, hcc.d);
    if (contactforces) cnt::halo(hpp, hff, hcc.d);

    /* send the forces back */ 
    
    download_ff(&e.pf);

    post_send_ff(&e.pf, &e.c);
    post_recv_ff(&e.c, &e.uf);

    wait_send_ff(&e.c);    
    wait_recv_ff(&e.c, &e.uf);

    unpack_ff(&e.uf, &e.p, nw, /**/ fw);
}

