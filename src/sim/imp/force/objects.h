void forces_cnt(int nw, PaWrap *pw, FoWrap *fw) {
    cnt::build_cells(nw, pw, /**/ &rs::c);
    cnt::bulk(&rs::c, nw, pw, fw);
}

void forces_fsi(fsi::SolventWrap *w_s, int nw, PaWrap *pw, FoWrap *fw) {
    fsi::bind(*w_s);
    fsi::bulk(nw, pw, fw);
}

void forces_objects(Flu *f, Rbc *r, Rig *s) {
    fsi::SolventWrap w_s;
    Cloud cloud;
    PaWrap pw[MAX_OBJ_TYPES];
    FoWrap fw[MAX_OBJ_TYPES];
    int nw = 0;
    
    if (solids0) {
        pw[nw] = {s->q.n, s->q.pp};
        fw[nw] = {s->q.n, s->ff};
        ++nw;
    }
    if (rbcs) {
        pw[nw] = {r->q.n, r->q.pp};
        fw[nw] = {r->q.n, r->ff};
        ++nw;
    }

    if (!nw) return;

    /* Prepare and send the data */
    
    using namespace rs;
    build_map(nw, pw, /**/ &e.p);
    pack(nw, pw, /**/ &e.p);
    download(nw, /**/ &e.p);

    UC(post_send(&e.p, &e.c));
    UC(post_recv(&e.c, &e.u));

    /* bulk interactions */
    
    ini_cloud(f->q.pp, &cloud);
    if (multi_solvent) ini_cloud_color(f->q.cc, &cloud);

    w_s.pp = f->q.pp;
    w_s.c  = cloud;
    w_s.ff = f->ff;
    w_s.n  = f->q.n;
    w_s.starts = f->q.cells.starts;

    if (contactforces) forces_cnt(nw, pw, fw);
    if (fsiforces)     forces_fsi(&w_s, nw, pw, fw);

    /* recv data and halo interactions  */

    wait_send(&e.c);
    wait_recv(&e.c, &e.u);

    int26 hcc = get_counts(&e.u);
    Pap26 hpp = upload_shift(&e.u);
    Fop26 hff = reini_ff(&e.u, &e.pf);

    if (fsiforces)     fsi::halo(hpp, hff, hcc.d);
    if (contactforces) cnt::halo(&rs::c, nw, pw, fw, hpp, hff, hcc.d);

    /* send the forces back */ 
    
    download_ff(&e.pf);

    UC(post_send_ff(&e.pf, &e.c));
    UC(post_recv_ff(&e.c, &e.uf));

    wait_send_ff(&e.c);    
    wait_recv_ff(&e.c, &e.uf);

    unpack_ff(&e.uf, &e.p, nw, /**/ fw);
}

