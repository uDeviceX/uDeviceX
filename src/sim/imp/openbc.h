void apply_inflow(Inflow *i, Flu *f) {
    create_pp(i, &f->q.n, f->q.pp);
}

void mark_outflow(const Flu *f, Outflow *o) {
    filter_particles(f->q.n, f->q.pp, /**/ o);
    download_ndead(o);
}

void mark_outflowden(const Flu *f, const DContMap *m, /**/ DCont *d) {
    const int *ss, *cc;
    ss = f->q.cells.starts;
    cc = f->q.cells.counts;

    filter_particles(m, ss, cc, /**/ d);
    download_ndead(d);
}
