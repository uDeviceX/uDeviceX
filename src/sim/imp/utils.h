static double compute_volume_rbc(MPI_Comm comm, const Rbc *r) {
    double loc, tot, V0;
    long nc;
    nc = r->q.nc;
    V0 = rbc_params_get_tot_volume(r->params);

    tot = 0;
    loc = nc * V0;
    MC(m::Allreduce(&loc, &tot, 1, MPI_DOUBLE, MPI_SUM, comm));
    
    return tot;
}

static void compute_hematocrit(const Sim *s) {
    const Opt *opt = &s->opt;
    double Vdomain, Vrbc, Ht;
    if (!opt->rbc) return;

    if (opt->wall) {
        enum {NSAMPLES = 100000};
        Vdomain = sdf_compute_volume(s->cart, s->params.L, s->wall.sdf, NSAMPLES);
    }
    else {
        const Coords *c = s->coords;
        Vdomain = xdomain(c) * ydomain(c) * zdomain(c);
    }

    Vrbc = compute_volume_rbc(s->cart, &s->rbc);
    
    Ht = Vrbc / Vdomain;

    msg_print("Geometry volume: %g", Vdomain);
    msg_print("Hematocrit: %g", Ht);
}
