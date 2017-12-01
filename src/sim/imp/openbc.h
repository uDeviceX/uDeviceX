void apply_inflow(Inflow *i, Flu *f) {
    create_pp(i, &f->q.n, f->q.pp);
}

void mark_outflow(const Flu *f, Outflow *o) {
    filter_particles(f->q.n, f->q.pp, /**/ o);
    download_ndead(o);
}
