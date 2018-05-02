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
    const Opt *opt = &s->opt;

    lp.pp = q->pp;

    if (opt->denoutflow) {
        lp.deathlist = den_get_deathlist(s->denoutflow);
        ndead        = den_get_ndead(s->denoutflow);
    }
    else if (opt->outflow) {
        lp.deathlist = outflow_get_deathlist(s->outflow);
        ndead        = outflow_get_ndead    (s->outflow);
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
}
