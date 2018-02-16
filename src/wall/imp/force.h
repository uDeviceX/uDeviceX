static WallForce get_wa(Wvel_v wv, Sdf *sdf, const WallQuants *q, const WallTicket *t, int n) {
    WallForce wa; /* local wall data */

    sdf_to_view(sdf, &wa.sdf_v);
    wa.start  = t->texstart;
    wa.pp     = t->texpp;
    wa.n      = q->n;
    wa.L      = q->L;

    return wa;
}

void wall_force(const PairParams *params, Wvel_v wv, const Coords *c, Sdf *sdf, const WallQuants *q, const WallTicket *t, int n, const PaArray *parray, const FoArray *farray) {
    WallForce wa;
    wa = get_wa(wv, sdf, q, t, n);

    wall_force_apply(params, wv, c, parray, n, t->rnd, wa, /**/ farray);
}
