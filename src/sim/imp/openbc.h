static void apply_inflow(float kBT, float dt, Inflow *i, Flu *f) {
    if (f->q.colors)
        UC(inflow_create_pp_cc(kBT, dt, RED_COLOR, i, &f->q.n, f->q.pp, f->q.cc));
    else
        UC(inflow_create_pp(kBT, dt, i, &f->q.n, f->q.pp));
}

static void mark_outflow(const Flu *f, Outflow *o) {
    UC(filter_particles(f->q.n, f->q.pp, /**/ o));
    UC(download_ndead(o));
}

static void mark_outflowden(const Flu *f, const DContMap *m, /**/ DCont *d) {
    const int *ss, *cc;
    int n;
    n = f->q.n;
    ss = f->q.cells.starts;
    cc = f->q.cells.counts;

    UC(den_reset(n, /**/ d));
    UC(den_filter_particles(m, ss, cc, /**/ d));
    UC(den_download_ndead(/**/ d));
}
