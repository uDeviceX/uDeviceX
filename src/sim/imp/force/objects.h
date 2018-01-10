void forces_cnt(ObjInter *oi, int nw, PaWrap *pw, FoWrap *fw) {
    cnt_build_cells(nw, pw, /**/ oi->cnt);
    cnt_bulk(oi->cnt, nw, pw, fw);
}

void forces_fsi(ObjInter *oi, fsi::SolventWrap *w_s, int nw, PaWrap *pw, FoWrap *fw) {
    fsi::bind(*w_s, &oi->fsi);
    fsi::bulk(&oi->fsi, nw, pw, fw);
}

void forces_objects(Sim *sim) {
    fsi::SolventWrap w_s;
    Cloud cloud;
    PaWrap pw[MAX_OBJ_TYPES];
    FoWrap fw[MAX_OBJ_TYPES];
    int nw = 0;
    ObjInter *oi = &sim->objinter;
    Flu *f = &sim->flu;
    Rbc *r = &sim->rbc;
    Rig *s = &sim->rig;
    ObjExch *e = &oi->e;
    
    if (sim->solids0) {
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
    
    build_map(nw, pw, /**/ &e->p);
    pack(nw, pw, /**/ &e->p);
    download(nw, /**/ &e->p);

    UC(post_send(&e->p, &e->c));
    UC(post_recv(&e->c, &e->u));

    /* bulk interactions */
    
    ini_cloud(f->q.pp, &cloud);
    if (multi_solvent) ini_cloud_color(f->q.cc, &cloud);

    w_s.c  = cloud;
    w_s.ff = f->ff;
    w_s.n  = f->q.n;
    w_s.starts = f->q.cells.starts;

    if (contactforces) forces_cnt(oi, nw, pw, fw);
    if (fsiforces)     forces_fsi(oi, &w_s, nw, pw, fw);

    /* recv data and halo interactions  */

    wait_send(&e->c);
    wait_recv(&e->c, &e->u);

    int26 hcc = get_counts(&e->u);
    Pap26 hpp = upload_shift(&e->u);
    Fop26 hff = reini_ff(&e->u, &e->pf);

    if (fsiforces)     fsi::halo(&oi->fsi, hpp, hff, hcc.d);
    if (contactforces) cnt_halo(oi->cnt, nw, pw, fw, hpp, hff, hcc.d);

    /* send the forces back */ 
    
    download_ff(&e->pf);

    UC(post_send_ff(&e->pf, &e->c));
    UC(post_recv_ff(&e->c, &e->uf));

    wait_send_ff(&e->c);    
    wait_recv_ff(&e->c, &e->uf);

    unpack_ff(&e->uf, &e->p, nw, /**/ fw);
}

