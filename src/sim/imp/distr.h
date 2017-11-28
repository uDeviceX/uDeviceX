/* the following functions will need to be splitted in the future 
   for performance reasons */

void distribute_flu(Flu *f) {
    flu::Quants *q = &f->q;
    FluDistr *d = &f->d;
    
    build_map(q->n, q->pp, /**/ &d->p);
    pack(q, /**/ &d->p);
    download(q->n, /**/ &d->p);

    UC(post_send(&d->p, &d->c));
    UC(post_recv(&d->c, &d->u));

    distr::flu::bulk(/**/ q);
    
    wait_send(&d->c);
    wait_recv(&d->c, &d->u);
    
    unpack(/**/ &d->u);
    
    halo(&d->u, /**/ q);
    gather(&d->p, &d->u, /**/ q);

    dSync();
}

void distribute_rbc() {
    using namespace r;

    build_map(q.nc, q.nv, q.pp, /**/ &d.p);
    pack(&q, /**/ &d.p);
    download(/**/&d.p);

    UC(post_send(&d.p, &d.c));
    UC(post_recv(&d.c, &d.u));

    unpack_bulk(&d.p, /**/ &q);

    wait_send(&d.c);
    wait_recv(&d.c, &d.u);

    unpack_halo(&d.u, /**/ &q);
    dSync();
}

void distribute_rig() {
    using namespace s;
    int nv = q.nv;

    build_map(q.ns, q.ss, /**/ &d.p);
    pack(q.ns, nv, q.ss, q.i_pp, /**/ &d.p);
    download(/**/&d.p);

    UC(post_send(&d.p, &d.c));
    UC(post_recv(&d.c, &d.u));

    unpack_bulk(&d.p, /**/ &q);
    
    wait_send(&d.c);
    wait_recv(&d.c, &d.u);

    unpack_halo(&d.u, /**/ &q);

    q.n = q.ns * q.nps;
    rig::generate(q.ns, q.ss, q.nps, q.rr0, /**/ q.pp);
    dSync();
}
