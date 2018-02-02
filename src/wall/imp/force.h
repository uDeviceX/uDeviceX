static WallForce get_wa(Wvel_v wv, Sdf *sdf, const WallQuants *q, const WallTicket *t, int n) {
    WallForce wa; /* local wall data */

    sdf_to_view(sdf, &wa.sdf_v);
    wa.start  = t->texstart;
    wa.pp     = t->texpp;
    wa.n      = q->n;
    wa.L      = q->L;

    return wa;
}

void wall_force(const PairParams *params, Wvel_v wv, const Coords *c, Sdf *sdf, const WallQuants *q, const WallTicket *t, int n, Cloud cloud, Force *ff) {
    WallForce wa;
    wa = get_wa(wv, sdf, q, t, n);

    wall_force_apply(params, wv, c, cloud, n, t->rnd, wa, /**/ ff);
}

void wall_force_color(const PairParams *params, Wvel_v wv, const Coords *c, Sdf *sdf, const WallQuants *q, const WallTicket *t, int n, Cloud cloud, Force *ff) {
    WallForce wa;
    wa = get_wa(wv, sdf, q, t, n);

    wall_force_apply_color(params, wv, c, cloud, n, t->rnd, wa, /**/ ff);
}
