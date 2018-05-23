static WallForce get_wa(const Sdf *sdf, const WallQuants *q, const WallTicket *t, int n) {
    WallForce wa; /* local wall data */

    sdf_get_view(sdf, &wa.sdf_v);
    wa.start  = t->texstart;
    wa.pp     = t->texpp;
    wa.n      = q->n;
    wa.L      = q->L;

    return wa;
}

void wall_force(const PairParams *params, const WvelStep *wv, const Coords *c, const Sdf *sdf,
                const WallQuants *q, const WallTicket *t, int n, const PaArray *parray, const FoArray *farray) {
    WallForce wa;
    wa = get_wa(sdf, q, t, n);

    UC(wall_force_apply(params, wv, c, parray, n, t->rnd, wa, /**/ farray));
}

void wall_force_adhesion(const PairParams *params, const WvelStep *wv, const Coords *c, const Sdf *sdf,
                         const WallQuants *q, const WallTicket *t, int n, const PaArray *parray, const FoArray *farray) {
    WallForce wa;
    wa = get_wa(sdf, q, t, n);

    UC(wall_force_adhesion_apply(params, wv, c, parray, n, t->rnd, wa, /**/ farray));
}

void wall_repulse(const Sdf *sdf, long n, const PaArray *pa, const FoArray *fa) {
    Sdf_v sdf_v;
    sdf_get_view(sdf, &sdf_v);
    UC(wall_force_repulse(sdf_v, n, pa, /**/ fa));
}
