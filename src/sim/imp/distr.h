/* the following functions will need to be splitted in the future 
   for performance reasons */

void distribute_flu(Sim *s) {
    PartList lp;
    flu::Quants *q = &s->flu.q;
    FluDistr *d = &s->flu.d;
    int ndead;
    
    lp.pp        = q->pp;

    if (s->opt.denoutflow) {
        lp.deathlist = den_get_deathlist(s->denoutflow);
        ndead        = den_get_ndead(s->denoutflow);
    }
    else if (s->opt.outflow) {
        lp.deathlist = get_deathlist(s->outflow);
        ndead = get_ndead(s->outflow);
    } else {
        lp.deathlist = NULL;
        ndead = 0;
    }

    // printf("n = %d\n", q->n);
    
    dflu_build_map(q->n, lp, /**/ &d->p);
    dflu_pack(q, /**/ &d->p);
    dflu_download(/**/ &d->p);

    UC(dflu_post_send(&d->p, &d->c));
    UC(dflu_post_recv(&d->c, &d->u));

    distr::flu::dflu_bulk(lp, /**/ q);
    
    dflu_wait_send(&d->c);
    dflu_wait_recv(&d->c, &d->u);
    
    dflu_unpack(/**/ &d->u);
    
    dflu_halo(&d->u, /**/ q);
    dflu_gather(ndead, &d->p, &d->u, /**/ q);

    dSync();
}

void distribute_rbc(Rbc *r) {
    rbc::Quants *q = &r->q;
    RbcDistr *d    = &r->d;
    
    build_map(q->nc, q->nv, q->pp, /**/ &d->p);
    pack(q, /**/ &d->p);
    download(/**/&d->p);

    UC(post_send(&d->p, &d->c));
    UC(post_recv(&d->c, &d->u));

    unpack_bulk(&d->p, /**/ q);

    wait_send(&d->c);
    wait_recv(&d->c, &d->u);

    unpack_halo(&d->u, /**/ q);
    dSync();
}

void distribute_rig(Rig *s) {
    rig::Quants *q = &s->q;
    RigDistr    *d = &s->d;
    int nv = q->nv;

    build_map(q->ns, q->ss, /**/ &d->p);
    pack(q->ns, nv, q->ss, q->i_pp, /**/ &d->p);
    download(/**/&d->p);

    UC(post_send(&d->p, &d->c));
    UC(post_recv(&d->c, &d->u));

    unpack_bulk(&d->p, /**/ q);
    
    wait_send(&d->c);
    wait_recv(&d->c, &d->u);

    unpack_halo(&d->u, /**/ q);

    q->n = q->ns * q->nps;
    rig::generate(q->ns, q->ss, q->nps, q->rr0, /**/ q->pp);
    dSync();
}
