void apply_inflow(float dt0, Inflow *i, Flu *f) {
    if (multi_solvent)
        UC(inflow_create_pp_cc(dt0, RED_COLOR, i, &f->q.n, f->q.pp, f->q.cc));
    else
        UC(inflow_create_pp(dt0, i, &f->q.n, f->q.pp));
}

void mark_outflow(const Flu *f, Outflow *o) {
    UC(filter_particles(f->q.n, f->q.pp, /**/ o));
    UC(download_ndead(o));
}

void mark_outflowden(const Flu *f, const DContMap *m, /**/ DCont *d) {
    const int *ss, *cc;
    int n;
    n = f->q.n;
    ss = f->q.cells.starts;
    cc = f->q.cells.counts;

    UC(den_reset(n, /**/ d));
    UC(den_filter_particles(m, ss, cc, /**/ d));
    UC(den_download_ndead(/**/ d));
}
