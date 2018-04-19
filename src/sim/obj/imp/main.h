static void distribute_mbr(Mbr *m) {
    RbcQuants *q = &m->q;
    MbrDistr  *d = &m->d;

    UC(drbc_build_map(q->nc, q->nv, q->pp, /**/ d->p));
    UC(drbc_pack(q, /**/ d->p));
    UC(drbc_download(/**/d->p));

    UC(drbc_post_send(d->p, d->c));
    UC(drbc_post_recv(d->c, d->u));

    UC(drbc_unpack_bulk(d->p, /**/ q));

    UC(drbc_wait_send(d->c));
    UC(drbc_wait_recv(d->c, d->u));

    UC(drbc_unpack_halo(d->u, /**/ q));
    dSync();
}

static void distribute_rig(Rig *r) {
    RigQuants *q = &r->q;
    RigDistr  *d = &r->d;
    int nv = q->nv;
    
    UC(drig_build_map(q->ns, q->ss, /**/ d->p));
    UC(drig_pack(q->ns, nv, q->ss, q->i_pp, /**/ d->p));
    UC(drig_download(/**/d->p));

    UC(drig_post_send(d->p, d->c));
    UC(drig_post_recv(d->c, d->u));

    UC(drig_unpack_bulk(d->p, /**/ q));

    UC(drig_wait_send(d->c));
    UC(drig_wait_recv(d->c, d->u));

    UC(drig_unpack_halo(d->u, /**/ q));

    q->n = q->ns * q->nps;
    UC(rig_generate(q->ns, q->ss, q->nps, q->rr0, /**/ q->pp));
    dSync();
}

void objects_distribute (Objects *obj) {
    if (obj->mbr) UC(distribute_mbr(obj->mbr));
    if (obj->rig) UC(distribute_rig(obj->rig));
}
