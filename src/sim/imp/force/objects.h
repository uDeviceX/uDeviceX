void forces_cnt(ObjInter *oi, int nw, PaWrap *pw, FoWrap *fw) {
    cnt_build_cells(nw, pw, /**/ oi->cnt);
    cnt_bulk(oi->cnt, nw, pw, fw);
}

void forces_fsi(ObjInter *oi, int nw, PaWrap *pw, FoWrap *fw) {
    fsi_bulk(oi->fsi, nw, pw, fw);
}

void forces_objects(Sim *sim) {
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
    
    eobj_build_map(nw, pw, /**/ &e->p);
    eobj_pack(nw, pw, /**/ &e->p);
    eobj_download(nw, /**/ &e->p);

    UC(eobj_post_send(&e->p, &e->c));
    UC(eobj_post_recv(&e->c, &e->u));

    /* bulk interactions */
    
    ini_cloud(f->q.pp, &cloud);
    if (multi_solvent) ini_cloud_color(f->q.cc, &cloud);

    if (fsiforces)
        fsi_bind_solvent(cloud, f->ff, f->q.n, f->q.cells.starts, /**/ oi->fsi);

    if (contactforces) forces_cnt(oi, nw, pw, fw);
    if (fsiforces)     forces_fsi(oi, nw, pw, fw);

    /* recv data and halo interactions  */

    UC(eobj_wait_send(&e->c));
    UC(eobj_wait_recv(&e->c, &e->u));

    int26 hcc = eobj_get_counts(&e->u);
    Pap26 hpp = eobj_upload_shift(&e->u);
    Fop26 hff = eobj_reini_ff(&e->u, &e->pf);

    if (fsiforces)     fsi_halo(oi->fsi, hpp, hff, hcc.d);
    if (contactforces) cnt_halo(oi->cnt, nw, pw, fw, hpp, hff, hcc.d);

    /* send the forces back */ 
    
    eobj_download_ff(&e->pf);

    UC(eobj_post_send_ff(&e->pf, &e->c));
    UC(eobj_post_recv_ff(&e->c, &e->uf));

    UC(eobj_wait_send_ff(&e->c));
    UC(eobj_wait_recv_ff(&e->c, &e->uf));

    eobj_unpack_ff(&e->uf, &e->p, nw, /**/ fw);
}

