/* the following functions will need to be splitted in the future
   for performance reasons */

static void log_and_fail(Coords *c, DFluStatus *s, FluQuants *q) {
    unsigned int time;
    time = 10;
    UC(dflu_status_log(s));
    UC(flu_txt_dump(c, q));
    msg_print("sleep for %d seconds", time);
    os_sleep(time); /* hope all ranks dump */
    ERR("dflu_download failed");
}
void distribute_flu(Sim *s) {
    PartList lp;
    FluQuants *q = &s->flu.q;
    FluDistr *d = &s->flu.d;
    int ndead;

    NVTX_PUSH("distr_flu");
    
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
    UC(dflu_build_map(q->n, lp, /**/ d->p));
    UC(dflu_pack(q, /**/ d->p));

    UC(dflu_download(/**/ d->p, d->s));
    if (!dflu_status_success(d->s))
        UC(log_and_fail(s->coords, d->s, q));

    UC(dflu_post_send(d->p, d->c));
    UC(dflu_post_recv(d->c, d->u));

    UC(dflu_bulk(lp, /**/ q));

    UC(dflu_wait_send(d->c));
    UC(dflu_wait_recv(d->c, d->u));

    UC(dflu_unpack(/**/ d->u));

    UC(dflu_halo(d->u, /**/ q));
    UC(dflu_gather(ndead, d->p, d->u, /**/ q));

    dSync();

    NVTX_POP();
}

void distribute_rbc(Rbc *r) {
    RbcQuants *q = &r->q;
    RbcDistr  *d = &r->d;

    NVTX_PUSH("distr_rbc");

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

    NVTX_POP();
}

void distribute_rig(Rig *s) {
    RigQuants *q = &s->q;
    RigDistr  *d = &s->d;
    int nv = q->nv;

    NVTX_PUSH("distr_rig");
    
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
    rig_generate(q->ns, q->ss, q->nps, q->rr0, /**/ q->pp);
    dSync();

    NVTX_POP();
}
