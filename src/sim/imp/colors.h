/* set colors of particles according to the RBCs */

void gen_colors(const Rbc *r, Colorer *c, Flu *f) {
    int nm, nv, n, nmhalo;

    n  = f->q.n;
    nm = r->q.nc;
    nv = r->q.nv;

    UC(emesh_build_map(nm, nv, r->q.pp, /**/ c->e.p));
    UC(emesh_pack(nv, r->q.pp, /**/ c->e.p));
    UC(emesh_download(c->e.p));

    UC(emesh_post_send(c->e.p, c->e.c));
    UC(emesh_post_recv(c->e.c, c->e.u));

    if (nm*nv && n > 0)
        aD2D(c->pp, r->q.pp, nm*nv);

    UC(emesh_wait_send(c->e.c));
    UC(emesh_wait_recv(c->e.c, c->e.u));

    UC(emesh_unpack(nv, c->e.u, /**/ &nmhalo, c->pp + nm * nv));
    nm += nmhalo;

    /* compute extents */
    if (n > 0) {
        UC(minmax(c->pp, nv, nm, /**/ c->minext, c->maxext));
        UC(collision_get_colors(f->q.pp, f->q.n, c->pp, r->tri, nv, nm,
                                c->minext, c->maxext, /**/ f->q.cc));
    }
}

void recolor_flux(const Coords *c, const Recolorer *opt, Flu *f) {
    int3 L = make_int3(xs(c), ys(c), zs(c));
    if (opt->flux_active)
        UC(color_linear_flux(c, L, opt->flux_dir, RED_COLOR, f->q.n, f->q.pp, /**/ f->q.cc));
}
