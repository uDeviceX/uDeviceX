/* set colors of particles according to the RBCs */

void gen_colors(const Rbc *r, Colorer *c, Flu *f) {
    int nm, nv, nmhalo;
    const int4 *tri;
    
    nm = r->q.nc;
    nv = r->q.nv;

    emesh_build_map(nm, nv, r->q.pp, /**/ c->e.p);
    emesh_pack(nv, r->q.pp, /**/ c->e.p);
    emesh_download(c->e.p);

    UC(emesh_post_send(c->e.p, c->e.c));
    UC(emesh_post_recv(c->e.c, c->e.u));

    if (nm * nv) CC(d::MemcpyAsync(c->pp, r->q.pp, nm * nv * sizeof(Particle), D2D));
    
    UC(emesh_wait_send(c->e.c));
    UC(emesh_wait_recv(c->e.c, c->e.u));

    emesh_unpack(nv, c->e.u, /**/ &nmhalo, c->pp + nm * nv);
    nm += nmhalo;

    /* compute extents */
    UC(minmax(c->pp, nv, nm, /**/ c->minext, c->maxext));

    tri = area_volume_tri(r->q.area_volume);
    UC(collision_get_colors(f->q.pp, f->q.n,
                            c->pp, tri,
                            r->q.nt, nv, nm,
                            c->minext, c->maxext, /**/ f->q.cc));
}

void recolor_flux(const Coords *c, Flu *f) {
    int3 L = make_int3(xs(c), ys(c), zs(c));
    if (RECOLOR_FLUX)
        recolor::linear_flux(c, L, COL_FLUX_DIR, RED_COLOR, f->q.n, f->q.pp, /**/ f->q.cc);
}
