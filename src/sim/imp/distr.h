/* the following functions will need to be splitted in the future 
   for performance reasons */

void distribute_flu(Sim *s) {
    PartList lp;
    flu::FluQuants *q = &s->flu.q;
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
    
    dflu_build_map(q->n, lp, /**/ d->p);
    dflu_pack(q, /**/ d->p);
    dflu_download(/**/ d->p);

    UC(dflu_post_send(d->p, d->c));
    UC(dflu_post_recv(d->c, d->u));

    dflu_bulk(lp, /**/ q);
    
    dflu_wait_send(d->c);
    dflu_wait_recv(d->c, d->u);
    
    dflu_unpack(/**/ d->u);
    
    dflu_halo(d->u, /**/ q);
    dflu_gather(ndead, d->p, d->u, /**/ q);

    dSync();
}

void distribute_rbc(Rbc *r) {
    rbc::Quants *q = &r->q;
    RbcDistr *d    = &r->d;
    
    drbc_build_map(q->nc, q->nv, q->pp, /**/ d->p);
    drbc_pack(q, /**/ d->p);
    drbc_download(/**/d->p);

    UC(drbc_post_send(d->p, d->c));
    UC(drbc_post_recv(d->c, d->u));

    drbc_unpack_bulk(d->p, /**/ q);

    drbc_wait_send(d->c);
    drbc_wait_recv(d->c, d->u);

    drbc_unpack_halo(d->u, /**/ q);
    dSync();
}

void distribute_rig(Rig *s) {
    rig::Quants *q = &s->q;
    RigDistr    *d = &s->d;
    int nv = q->nv;

    drig_build_map(q->ns, q->ss, /**/ d->p);
    drig_pack(q->ns, nv, q->ss, q->i_pp, /**/ d->p);
    drig_download(/**/d->p);

    UC(drig_post_send(d->p, d->c));
    UC(drig_post_recv(d->c, d->u));

    drig_unpack_bulk(d->p, /**/ q);
    
    UC(drig_wait_send(d->c));
    UC(drig_wait_recv(d->c, d->u));

    drig_unpack_halo(d->u, /**/ q);

    q->n = q->ns * q->nps;
    rig::generate(q->ns, q->ss, q->nps, q->rr0, /**/ q->pp);
    dSync();
}
