void wall_force(Wvel_v wv, const Coords *c, Sdf *sdf, const WallQuants *q, const WallTicket *t, int n, Cloud cloud, Force *ff) {
    WallForce wa; /* local wall data */

    sdf_to_view(sdf, &wa.sdf_v);
    wa.start  = t->texstart;
    wa.pp     = t->texpp;
    wa.n      = q->n;

    wall_force_apply(wv, c, cloud, n, t->rnd, wa, /**/ ff);
}
