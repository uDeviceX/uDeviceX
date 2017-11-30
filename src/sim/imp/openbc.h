void apply_inflow(Inflow *i, Flu *f) {
    create_pp(i, &f->q.n, f->q.pp);
}

void mark_outflow(const Flu *f, Outflow *o) {
    float3 normal, r;
    normal = make_float3(1, 0, 0);
    r = make_float3(XS/2-1, 0, 0);
    filter_particles_plane(normal, r, f->q.n, f->q.pp, /**/ o);
    download_ndead(o);
}
