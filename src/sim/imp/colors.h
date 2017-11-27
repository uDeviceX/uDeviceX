/* set colors of particles according to the RBCs */

void gen_colors(Colorer *c) {
    int nm, nv, nmhalo;
    nm = r::q.nc;
    nv = r::q.nv;

    build_map(nm, nv, r::q.pp, /**/ &c->e.p);
    pack(nv, r::q.pp, /**/ &c->e.p);
    download(&c->e.p);

    UC(post_send(&c->e.p, &c->e.c));
    UC(post_recv(&c->e.c, &c->e.u));

    if (nm * nv) CC(d::MemcpyAsync(c->pp, r::q.pp, nm * nv * sizeof(Particle), D2D));
    
    wait_send(&c->e.c);
    wait_recv(&c->e.c, &c->e.u);

    unpack(nv, &c->e.u, /**/ &nmhalo, c->pp + nm * nv);
    nm += nmhalo;

    /* compute extents */
    minmax(c->pp, nv, nm, /**/ c->minext, c->maxext);

    /* tmp texture object; TODO: make a ticket? */
    Texo<float2> texvert;
    TE(&texvert, (float2*) c->pp, 3 * nm * nv);

    collision::get_colors(o::q.pp, o::q.n, texvert, r::q.tri,
                          r::q.nt, nv, nm, c->minext, c->maxext, /**/ o::q.cc);
    destroy(&texvert);
}

void recolor_flux() {
    if (RECOLOR_FLUX)
        recolor::flux(COL_FLUX_DIR, RED_COLOR, o::q.n, o::q.pp, /**/ o::q.cc);
}
